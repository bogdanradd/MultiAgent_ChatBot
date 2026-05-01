from pathlib import Path
from src.ingestion import ingest
from src.vectorstore import get_store, get_all_ingested_docs
from src import router


def show_banner():
    """Display welcome banner."""
    print("=" * 70)
    print("    FINANCIAL DOCS RAG CHATBOT")
    print("=" * 70)


def show_documents():
    """Display all ingested documents."""
    docs = get_all_ingested_docs()
    print("\nINGESTED DOCUMENTS:")
    print("-" * 70)
    if docs:
        for i, (doc_id, info) in enumerate(docs.items(), 1):
            print(f"{i}. {info['source']}")
            print(f"   Doc ID: {doc_id}")
            print(f"   Chunks: {info['chunks']}")
            print()
    else:
        print("   No documents ingested yet.")
        print()
    return docs


def load_api_data_menu():
    """Interactive API data loading menu."""
    from src.api_sources import ingest_api_data

    print("\nLOAD API DATA")
    print("-" * 70)
    print("1. Stock Price Data (Yahoo Finance)")
    print("2. Company Information (Yahoo Finance)")
    print("3. Back to main menu")

    choice = input("\nSelect data type [1-3]: ").strip()

    if choice == "3":
        return

    if choice == "1":
        symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
        days = input("Enter number of days (default 30): ").strip()
        days = int(days) if days.isdigit() else 30

        try:
            print(f"\nFetching stock data for {symbol}...")
            doc_id = ingest_api_data('stock', symbol, days=days)
            store = get_store()
            chunks = store.get(where={"doc_id": doc_id})
            n_chunks = len(chunks["ids"])
            print(f"[SUCCESS] Ingested {symbol} stock data -> {n_chunks} chunks (doc_id: {doc_id})")
        except Exception as e:
            print(f"[ERROR] Failed to ingest stock data: {e}")

    elif choice == "2":
        symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()

        try:
            print(f"\nFetching company info for {symbol}...")
            doc_id = ingest_api_data('company_info', symbol)
            store = get_store()
            chunks = store.get(where={"doc_id": doc_id})
            n_chunks = len(chunks["ids"])
            print(f"[SUCCESS] Ingested {symbol} company info -> {n_chunks} chunks (doc_id: {doc_id})")
        except Exception as e:
            print(f"[ERROR] Failed to ingest company info: {e}")

    input("\nPress Enter to continue...")


def load_documents_menu():
    """Interactive document loading menu."""
    print("\nLOAD DOCUMENTS")
    print("-" * 70)
    print("Enter file path(s) to ingest (PDF or CSV)")
    print("You can enter multiple paths separated by commas")
    print("Type 'back' to return to main menu")

    while True:
        user_input = input("\nFile path(s): ").strip()

        if user_input.lower() == 'back':
            break

        if not user_input:
            continue

        paths = [p.strip() for p in user_input.split(',')]

        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                print(f"[ERROR] File not found: {path_str}")
                continue

            try:
                print(f"\nIngesting {path.name}...")
                doc_id = ingest(str(path))
                store = get_store()
                chunks = store.get(where={"doc_id": doc_id})
                n_chunks = len(chunks["ids"])
                print(f"[SUCCESS] Ingested {path.name} -> {n_chunks} chunks (doc_id: {doc_id})")
            except Exception as e:
                print(f"[ERROR] Error ingesting {path.name}: {e}")

        print("\nLoad more documents? (or type 'back')")


def chat_menu():
    """Interactive chat interface with document scope selection."""
    docs = get_all_ingested_docs()

    if not docs:
        print("\n[WARNING] No documents loaded. Please load documents first.")
        input("\nPress Enter to continue...")
        return

    print("\nCHAT INTERFACE")
    print("-" * 70)
    print("Select document scope:")
    print("0. All documents")

    doc_list = list(docs.items())
    for i, (doc_id, info) in enumerate(doc_list, 1):
        print(f"{i}. {info['source']} ({info['chunks']} chunks)")

    while True:
        choice = input("\nScope [0-{}]: ".format(len(doc_list))).strip()

        if not choice.isdigit():
            print("[ERROR] Please enter a number")
            continue

        choice_num = int(choice)
        if choice_num < 0 or choice_num > len(doc_list):
            print(f"[ERROR] Please enter a number between 0 and {len(doc_list)}")
            continue

        if choice_num == 0:
            doc_id = None
            scope_name = "All documents"
        else:
            doc_id = doc_list[choice_num - 1][0]
            scope_name = doc_list[choice_num - 1][1]['source']

        break

    print("\n" + "=" * 70)
    print(f"SCOPE: {scope_name}")
    print("=" * 70)
    print("\nThe router automatically picks the right agent for your message:")
    print("   - QA Agent - Answer questions from documents")
    print("   - Summarizer - Generate comprehensive summaries")
    print("   - MCQ Generator - Create practice questions")
    print("\nType 'exit' or 'back' to return to main menu")
    print("-" * 70)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'back', 'quit']:
                break

            print("\nProcessing...")
            response = router.handle(user_input, doc_id=doc_id)
            intent = response["intent"]
            result = response["result"]

            print("\n" + "=" * 70)

            if intent == "SUMMARIZE":
                print("AGENT: Summarizer (mistral:7b-instruct)")
                print("OPERATION: Map-reduce summarization")
                print("=" * 70)
                print(f"\n{result}")

            elif intent == "QA":
                print("AGENT: Q&A (llama3.1:8b)")
                print("OPERATION: Retrieval-based question answering")
                print("=" * 70)
                print(f"\n{result}")

            elif intent == "MCQ":
                print("AGENT: MCQ Generator (qwen2.5:7b-instruct)")
                print("OPERATION: Multiple-choice question generation")
                print("=" * 70)
                print()
                if isinstance(result, list):
                    for i, mcq in enumerate(result, 1):
                        print(f"Question {i}: {mcq.question}")
                        for j, option in enumerate(mcq.options):
                            print(f"   {chr(65+j)}. {option}")
                        print(f"\n   Answer: {chr(65+mcq.answer_index)}. {mcq.options[mcq.answer_index]}")
                        print(f"   Explanation: {mcq.explanation}\n")
                else:
                    print(f"{result}")

            print("=" * 70)

        except ValueError as e:
            print(f"\n[ERROR] {e}")
        except KeyboardInterrupt:
            print("\n\nReturning to main menu...")
            break
        except EOFError:
            break


def main_menu():
    """Interactive main menu."""
    show_banner()

    while True:
        docs = get_all_ingested_docs()

        print("\nMAIN MENU")
        print("-" * 70)
        print("1. Load Documents (PDF/CSV)")
        print("2. Load API Data (Stock/Company Info)")
        print("3. Chat (RAG + Agentic Workflow)")
        print("4. View Loaded Documents")
        print("5. Exit")

        if docs:
            print(f"\nCurrently loaded: {len(docs)} document(s)")
        else:
            print("\nNo documents loaded yet")

        choice = input("\nSelect option [1-5]: ").strip()

        if choice == "1":
            load_documents_menu()
        elif choice == "2":
            load_api_data_menu()
        elif choice == "3":
            chat_menu()
        elif choice == "4":
            show_documents()
            input("\nPress Enter to continue...")
        elif choice == "5":
            print("\nGoodbye!")
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main_menu()
