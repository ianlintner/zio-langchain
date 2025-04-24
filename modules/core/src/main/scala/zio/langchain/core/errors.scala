package zio.langchain.core

/**
 * Errors that can occur in the LangChain library.
 */
object errors {

  /**
   * Base trait for all errors in the LangChain library.
   */
  trait LangChainError extends Throwable

  /**
   * Error that occurs in an LLM.
   */
  case class LLMError(cause: Throwable) extends LangChainError {
    override def getMessage: String = cause.getMessage
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in a chain.
   */
  case class ChainError(cause: Throwable) extends LangChainError {
    override def getMessage: String = cause.getMessage
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in a tool.
   */
  case class ToolError(cause: Throwable, message: String = "") extends LangChainError {
    override def getMessage: String = if (message.isEmpty) cause.getMessage else message
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in an agent.
   */
  case class AgentError(cause: Throwable) extends LangChainError {
    override def getMessage: String = cause.getMessage
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in a retriever.
   */
  case class RetrieverError(cause: Throwable, message: String = "") extends LangChainError {
    override def getMessage: String = if (message.isEmpty) cause.getMessage else message
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in a memory.
   */
  case class MemoryError(cause: Throwable, message: String = "") extends LangChainError {
    override def getMessage: String = if (message.isEmpty) cause.getMessage else message
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in a document loader.
   */
  case class DocumentLoaderError(cause: Throwable) extends LangChainError {
    override def getMessage: String = cause.getMessage
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in a document parser.
   */
  case class DocumentParserError(cause: Throwable) extends LangChainError {
    override def getMessage: String = cause.getMessage
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in an embedding model.
   */
  case class EmbeddingError(cause: Throwable, message: String = "") extends LangChainError {
    override def getMessage: String = if (message.isEmpty) cause.getMessage else message
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in a vector store.
   */
  case class VectorStoreError(cause: Throwable) extends LangChainError {
    override def getMessage: String = cause.getMessage
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in a parser.
   */
  case class ParserError(cause: Throwable) extends LangChainError {
    override def getMessage: String = cause.getMessage
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in a prompt template.
   */
  case class PromptTemplateError(cause: Throwable) extends LangChainError {
    override def getMessage: String = cause.getMessage
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs in a configuration.
   */
  case class ConfigurationError(message: String) extends LangChainError {
    override def getMessage: String = message
  }

  /**
   * Error that occurs when executing a tool.
   */
  case class ToolExecutionError(message: String, cause: Throwable = null) extends LangChainError {
    override def getMessage: String = message
    override def getCause: Throwable = cause
  }
  
  /**
   * Error that occurs during output parsing.
   */
  case class OutputParsingError(cause: Throwable, message: String, rawText: Option[String] = None) extends LangChainError {
    override def getMessage: String = message
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs when loading a document.
   */
  case class DocumentLoadingError(cause: Throwable) extends LangChainError {
    override def getMessage: String = cause.getMessage
    override def getCause: Throwable = cause
  }

  /**
   * Error that occurs when parsing a document.
   */
  case class DocumentParsingError(cause: Throwable) extends LangChainError {
    override def getMessage: String = cause.getMessage
    override def getCause: Throwable = cause
  }

  /**
   * OpenAI API specific error helpers
   */
  object OpenAIError {
    // Create wrapper methods to convert OpenAI errors to LLMError
    def authenticationError(message: String): LLMError =
      LLMError(new RuntimeException(s"Authentication error: $message"))
    
    def rateLimitError(message: String): LLMError =
      LLMError(new RuntimeException(s"Rate limit exceeded: $message"))
    
    def serverError(message: String): LLMError =
      LLMError(new RuntimeException(s"OpenAI server error: $message"))
    
    def invalidRequestError(message: String): LLMError =
      LLMError(new RuntimeException(s"Invalid request: $message"))
    
    def timeoutError(message: String): LLMError =
      LLMError(new RuntimeException(s"Request timed out: $message"))
    
    def unknownError(cause: Throwable): LLMError =
      LLMError(cause)
  }

  /**
   * Pinecone API specific error helpers
   */
  object PineconeError {
    // Create wrapper methods to convert Pinecone errors to RetrieverError
    def authenticationError(message: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Authentication error: $message"), "Pinecone authentication failed")
    
    def rateLimitError(message: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Rate limit exceeded: $message"), "Pinecone rate limit exceeded")
    
    def serverError(message: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Pinecone server error: $message"), "Pinecone server error")
    
    def invalidRequestError(message: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Invalid request: $message"), "Invalid request to Pinecone API")
    
    def timeoutError(message: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Request timed out: $message"), "Pinecone request timed out")
    
    def indexNotFoundError(indexName: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Index not found: $indexName"), "Pinecone index not found")
    
    def dimensionMismatchError(expected: Int, actual: Int): RetrieverError =
      RetrieverError(new RuntimeException(s"Dimension mismatch: expected $expected, got $actual"),
                    "Vector dimension mismatch in Pinecone request")
    
    def unknownError(cause: Throwable): RetrieverError =
      RetrieverError(cause, "Unknown Pinecone error")
  }
  /**
   * PgVector API specific error helpers
   */
  object PgVectorError {
    // Create wrapper methods to convert PgVector errors to RetrieverError
    def connectionError(message: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Connection error: $message"), "PgVector connection failed")
    
    def queryError(message: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Query error: $message"), "PgVector query failed")
    
    def serverError(message: String): RetrieverError =
      RetrieverError(new RuntimeException(s"PgVector server error: $message"), "PgVector server error")
    
    def invalidRequestError(message: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Invalid request: $message"), "Invalid request to PgVector")
    
    def timeoutError(message: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Request timed out: $message"), "PgVector request timed out")
    
    def tableNotFoundError(tableName: String): RetrieverError =
      RetrieverError(new RuntimeException(s"Table not found: $tableName"), "PgVector table not found")
    
    def dimensionMismatchError(expected: Int, actual: Int): RetrieverError =
      RetrieverError(new RuntimeException(s"Dimension mismatch: expected $expected, got $actual"),
                    "Vector dimension mismatch in PgVector request")
    
    def unknownError(cause: Throwable): RetrieverError =
      RetrieverError(cause, "Unknown PgVector error")
  }
}