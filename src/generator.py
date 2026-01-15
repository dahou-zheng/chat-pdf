import sys
import functools
import openai
from collections import defaultdict
from heapq import nlargest
from typing import List, Dict, Callable, Any
from document_processor import _load_local_documents, split_documents_to_text_chunks
from vector_store import FaissManager
from config import (
    OPENAI_API_KEY,
    DEFAULT_MODEL,
    TEST_PDFS_DIR,
    DEFAULT_TOP_K,
)


def load_client(api_key: str = OPENAI_API_KEY) -> openai.OpenAI:
    """
    Initialize and return OpenAI client with error handling.
    """
    try:
        # Initialize the client
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing from configuration.")

        client = openai.OpenAI(api_key=api_key)

        # A "ping" check to verify connectivity/quota immediately
        client.models.list()

        return client

    except openai.APIConnectionError as e:
        print(f"Error: The server could not be reached. {e}")
        sys.exit(1)
    except openai.AuthenticationError as e:
        print(f"Error: Your OpenAI API key or token is invalid. {e}")
        sys.exit(1)
    except openai.RateLimitError as e:
        print(f"Error: You have hit your OpenAI rate limit or quota: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during OpenAI initialization: {e}")
        sys.exit(1)


client = load_client()


def handle_openai_errors(func: Callable) -> Callable:
    """
    Decorator to handle OpenAI API exceptions and network issues.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except openai.APIConnectionError as e:
            # Handles network issues (DNS, no internet, connection refused)
            raise ConnectionError(f"Could not connect to OpenAI API: {e}")
        except openai.APITimeoutError as e:
            # Handles cases where the request takes too long
            raise TimeoutError(f"OpenAI API request timed out: {e}")
        except openai.RateLimitError as e:
            # Handles 429 errors (Quota exceeded or too many requests)
            raise RuntimeError(f"Rate limit hit: {e}. Check your credits or throughput limits.")
        except openai.AuthenticationError as e:
            # Handles 401 errors (Invalid API Key)
            raise ValueError(f"Authentication failed: {e}")
        except openai.BadRequestError as e:
            # Handles 400 errors (Wrong model name, invalid parameters, etc.)
            raise ValueError(f"Invalid request to OpenAI: {e}")
        except openai.APIStatusError as e:
            # Handles 5xx errors (OpenAI server-side issues)
            raise RuntimeError(f"OpenAI server returned an error (Status {e.status_code}): {e.response}")
        except Exception as e:
            # Fallback for any other unexpected errors
            raise RuntimeError(f"An unexpected error occurred: {e}")
    return wrapper


@handle_openai_errors
def generate_query_reformulations(
        original_query: str,
        model: str = DEFAULT_MODEL,
        num_reformulations: int = 3,
        temperature: float = 0.8,
        max_tokens: int = 300
) -> List[str]:
    """
    Generate query reformulations using LLM

    Args:
        original_query: Original user query
        model: name of model to use
        num_reformulations: Number of reformulations to generate (default 3)
        temperature: Temperature parameter for diversity (default 0.8)
        max_tokens: Maximum tokens for the response (default 300)

    Returns:
        List of reformulated queries
    """

    system_prompt = ("You are a query reformulation assistant. Generate alternative phrasings "
                     "of the given query that would help retrieve relevant information.")

    user_prompt = f"""\
Given the following query, generate {num_reformulations} different reformulations that:
1. Express the same intent but use different wording
2. May use synonyms or related terms
3. Could be phrased as questions or statements
4. Help retrieve relevant information from a document search system

Original Query: {original_query}

Generate exactly {num_reformulations} reformulations, one per line, without numbering or bullets."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,  # prevents runaway costs
    )

    reformulations_text = response.choices[0].message.content.strip()

    # Parse reformulations (split by newlines and clean)
    reformulations = []
    for line in reformulations_text.strip().split('\n'):
        line = line.strip()
        # Remove numbering if present (e.g., "1. ", "- ", etc.)
        for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        if line and len(line) > 5:  # Filter out very short lines
            reformulations.append(line)
            if len(reformulations) == num_reformulations:
                break

    # Return exactly num_reformulations, or pad with original if needed
    while len(reformulations) < num_reformulations:
        reformulations.append(original_query)

    return reformulations


def reciprocal_rank_fusion(
        search_results_list: List[List[Dict]],
        k: int = 60,
        top_k: int = DEFAULT_TOP_K
) -> List[Dict]:
    """
    Apply Reciprocal Rank Fusion (RRF) to combine multiple search result lists

    Args:
        search_results_list: List of search result lists (each from a different query)
        k: RRF constant (default 60)
        top_k: Number of top results to return after reranking (default DEFAULT_TOP_K)

    Returns:
        Reranked list of results with combined scores
    """
    # Dictionary to store RRF scores: {chunk_id: rrf_score}
    rrf_scores = defaultdict(float)
    chunk_data = {}  # Store chunk data by ID

    # Process each search result list
    for results in search_results_list:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.get('id', None)
            if chunk_id:
                # RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)
                rrf_scores[chunk_id] += rrf_score

                # Store chunk data (use first occurrence or best score)
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = result
                else:
                    # Keep the one with better original score
                    if result.get('score', 0) > chunk_data[chunk_id].get('score', 0):
                        chunk_data[chunk_id] = result

    # get top k results by RRF score (descending)
    top_chunks = nlargest(top_k, rrf_scores.items(), key=lambda x: x[1])

    # Build final results with RRF scores
    final_results = []
    for chunk_id, rrf_score in top_chunks:
        result = chunk_data[chunk_id].copy()
        result['rrf_score'] = rrf_score
        result['score'] = rrf_score
        final_results.append(result)

    return final_results


def format_context(results: List[Dict]) -> str:
    """
    Format retrieved results into context string
    """
    context_parts = [f"[Chunk {i}] {r['text']}" for i, r in enumerate(results, 1)]
    context = "\n\n".join(context_parts)
    return context


@handle_openai_errors
def generate_answer(
        context_text: str,
        user_question: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1500
) -> str:
    """
    Generate answer based on context information

    Args:
        context_text: Context information (usually retrieved document chunks)
        user_question: User question
        model: name of the model to use, defaults to environment variable
        temperature: Temperature parameter, default 0.7
        max_tokens: Maximum tokens for the answer (curb the cost), default 1500

    Returns:
        Generated answer text
    """
    system_prompt = ("You are a professional Q&A assistant. "
                     "Please answer user questions accurately based on the provided context information.")

    user_prompt = f"""\
Context Information:
{context_text}

User Question: {user_question}

Requirements:
1. Only answer based on the provided context information, do not make up information
2. If there is no relevant information in the context, please clearly state so
3. Answers should be accurate, concise, and well-organized
4. You are encouraged to cite specific document sources

Please answer:"""

    # Call Open AI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,  # adjust based on your needs
    )

    answer = response.choices[0].message.content.strip()

    if not answer:
        raise ValueError("LLM returned empty answer")

    return answer


@handle_openai_errors
def condense_multi_turn_query(
        conversation_history: List[Dict[str, str]],
        current_question: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 200
) -> str:
    """
    Condense a multi-turn conversation into a single standalone retrieval query.

    Args:
        conversation_history: List of {"role": "user"/"assistant", "content": "..."}
                              Excludes the current user question.
        current_question: The latest user question.
        model: LLM model name
        temperature: Low temperature for determinism, default 0.2
        max_tokens: Token limit for safety

    Returns:
        A single condensed standalone query string
    """

    system_prompt = """You are a search query condensation assistant.
Your task is to rewrite the user's latest question into a SINGLE, \
standalone, explicit query suitable for document retrieval.


Rules:
1. Resolve all references (it, they, that, this, etc.) using conversation context
2. Preserve technical accuracy and intent
3. Do NOT answer the question
4. Do NOT add new facts not stated or implied
5. Optimize for semantic search, not chat
6. Output ONLY the rewritten query"""

    history_text = []
    for msg in conversation_history:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "").strip()
        if content:
            history_text.append(f"{role}: {content}")
    history_text = "\n".join(history_text)

    user_prompt = f"""\
Conversation History:
{history_text}

Latest User Question:
{current_question}

Standalone Search Query:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    condensed_query = response.choices[0].message.content.strip()

    if not condensed_query:
        raise ValueError("Failed to generate condensed query")

    return condensed_query


if __name__ == "__main__":
    # TODO: Make sure vector_store is functioning properly before running this test.
    index_manager = FaissManager()
    test_documents = _load_local_documents(TEST_PDFS_DIR)
    test_chunks = split_documents_to_text_chunks(test_documents)
    index_manager.add_chunks(test_chunks)

    # Test example
    test_question = "Why do language models follow instructions? Is Human feedback also reducing hallucination?"
    test_reformulations = generate_query_reformulations(test_question)

    # Search relevant content from FAISS
    print(f"Searching question and reformulated questions: {test_question} and {test_reformulations}")
    result_count = 0
    all_search_results = []
    initial_results = index_manager.search(test_question)
    result_count += len(initial_results)
    all_search_results.append(initial_results)
    for reformed_query in test_reformulations:
        reformed_results = index_manager.search(query=reformed_query)
        result_count += len(reformed_results)
        all_search_results.append(reformed_results)
    print(f"collect {result_count} text chunks from all searches")

    # rerank results with RRF
    reranked_results = reciprocal_rank_fusion(all_search_results, top_k=5)
    print(f"rerank to  get top {len(reranked_results)} text chunks")

    # Merge search results into context
    context = format_context(reranked_results)

    print(f"\nFound {len(reranked_results)} relevant chunks")
    print("\n" + "=" * 60)
    for result in reranked_results:
        print(f"\n{result['text']}\n")
    print("=" * 60)

    # Generate answer
    test_answer = generate_answer(context_text=context, user_question=test_question)

    print("\nQuestion:", test_question)
    print("\nAnswer:", test_answer)

    test_followup_question = ("So, hallucination reduction isn't the main progress, "
                              "then what is the major achievement of this training method?")
    conversation_history = [{"role": "user", "content": test_question}, {"role": "assistant", "content": test_answer}]
    condensed_query = condense_multi_turn_query(conversation_history, test_followup_question)
    print("\nOriginal Follow-up Question:", test_followup_question)
    print("\nCondensed Follow-up Query:", condensed_query)
    print("Now, input the condensed query to generate_query_reformulations function, this process can be repeated.")
    print("Thus, we have completed a multi-turn RAG with question reformulation, "
          "reciprocal_rank_fusion, and conversation history condensation.")
