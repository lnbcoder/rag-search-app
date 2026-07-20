"""Graph builder for LangGraph workflow"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes

class GraphBuilder:
    """Builds a LangGraph workflow for RAG"""

    def __init__(self, retriever, llm):
        """Initialize the graph builder with a retriever and LLM"""
        self.nodes = None
        self.graph = None

    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # create state graph
        builder = StateGraph(RAGState)
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)

        # set entry point
        builder.set_entry_point("retriever")

        # Add edges between nodes
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        #compile the graph
        self.graph = builder.compile()
        return self.graph
    

    def run(self, question: str) -> dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)