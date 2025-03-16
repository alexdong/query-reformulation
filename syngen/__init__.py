from syngen.triple_query import generate_triple_query

def main() -> None:
    """Main entry point for the syngen package."""
    print("[START] Generating a simple triple query from WikiData")
    query, metadata = generate_triple_query()
    print("\n" + "="*50)
    print(f"Generated Query: {query}")
    print("="*50)
    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print("="*50)

if __name__ == "__main__":
    main()
