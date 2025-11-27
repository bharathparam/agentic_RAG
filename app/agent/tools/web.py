import os
from langchain_community.tools.tavily_search import TavilySearchResults

class WebSearch:
    def __init__(self):
        self.tool = TavilySearchResults(max_results=3)

    def search(self, query: str) -> str:
        """Perform a web search."""
        try:
            results = self.tool.invoke({"query": query})
            # Format results into a single string
            formatted_results = []
            for res in results:
                formatted_results.append(f"Source: {res['url']}\nContent: {res['content']}")
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error performing web search: {str(e)}"
