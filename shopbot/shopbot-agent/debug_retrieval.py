# debug_retrieval.py
# Standalone retrieval debugger — no HTTP server, no timeout.
# Run: python debug_retrieval.py
# Add breakpoint() anywhere here or inside retrieval/ modules to step with pdb.
# VS Code: open this file, set gutter breakpoints, press F5 (uses .vscode/launch.json).

from dotenv import load_dotenv
load_dotenv()

from retrieval.multi_query import retrieve_multi

QUERY = "show me something festive and warm"

chunks, sub_queries = retrieve_multi(QUERY)

print(f"\nQuery: {QUERY}")
print(f"sub_queries: {sub_queries}")
print(f"chunks returned: {len(chunks)}\n")
for i, c in enumerate(chunks):
    print(f"[{i}] {c.text[:200]}")
    print()
