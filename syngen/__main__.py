import sys
from typing import List

from .knowledge_graph import generate_triple_query

def main() -> None:
    """Main entry point for the syngen package."""
    args = sys.argv[1:]
    
    if not args:
        # Default behavior - generate a query
        query, subqueries = generate_triple_query()
        print("\n" + "="*50)
        print(f"QUERY: {query}")
        print("SUBQUERIES:")
        for i, sq in enumerate(subqueries, 1):
            print(f"  {i}. {sq}")
        print("="*50)
    elif args[0] == "--help" or args[0] == "-h":
        print("Usage: python -m syngen [entity_type]")
        print("  entity_type: person, organization, place, book, movie (optional)")
    else:
        # Use the provided entity type
        entity_type = args[0]
        query, subqueries = generate_triple_query(entity_type)
        print("\n" + "="*50)
        print(f"QUERY: {query}")
        print("SUBQUERIES:")
        for i, sq in enumerate(subqueries, 1):
            print(f"  {i}. {sq}")
        print("="*50)

if __name__ == "__main__":
    main()
