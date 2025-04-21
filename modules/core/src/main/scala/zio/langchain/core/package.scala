package zio.langchain

/**
 * Core package containing the fundamental abstractions and types for ZIO LangChain.
 */
package object core {
  // Re-export error types
  type LangChainError = errors.LangChainError
  
  // Error re-exports for convenience
  val AgentError = zio.langchain.core.errors.AgentError
  val DocumentLoadingError = zio.langchain.core.errors.DocumentLoadingError
  val DocumentParsingError = zio.langchain.core.errors.DocumentParsingError
  val EmbeddingError = zio.langchain.core.errors.EmbeddingError
  val LLMError = zio.langchain.core.errors.LLMError
  val MemoryError = zio.langchain.core.errors.MemoryError
  val RetrieverError = zio.langchain.core.errors.RetrieverError
  val ToolExecutionError = zio.langchain.core.errors.ToolExecutionError
}