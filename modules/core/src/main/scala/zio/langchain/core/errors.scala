package zio.langchain.core

/**
 * Error types for ZIO LangChain.
 */
object errors:
  /**
   * Base trait for all errors in ZIO LangChain.
   */
  trait LangChainError extends Throwable
  /**
   * Error that occurs during LLM operations.
   *
   * @param cause The underlying cause of the error
   * @param message A descriptive error message
   */
  case class LLMError(
    cause: Throwable,
    message: String = "LLM error occurred"
  ) extends LangChainError:
    override def getMessage: String = s"$message: ${cause.getMessage}"
    override def getCause: Throwable = cause
  
  /**
   * Error that occurs during embedding operations.
   *
   * @param cause The underlying cause of the error
   * @param message A descriptive error message
   */
  case class EmbeddingError(
    cause: Throwable,
    message: String = "Embedding error occurred"
  ) extends LangChainError:
    override def getMessage: String = s"$message: ${cause.getMessage}"
    override def getCause: Throwable = cause
  
  /**
   * Error that occurs during document retrieval operations.
   *
   * @param cause The underlying cause of the error
   * @param message A descriptive error message
   */
  case class RetrieverError(
    cause: Throwable,
    message: String = "Retrieval error occurred"
  ) extends LangChainError:
    override def getMessage: String = s"$message: ${cause.getMessage}"
    override def getCause: Throwable = cause
  
  /**
   * Error that occurs during memory operations.
   *
   * @param cause The underlying cause of the error
   * @param message A descriptive error message
   */
  case class MemoryError(
    cause: Throwable,
    message: String = "Memory error occurred"
  ) extends LangChainError:
    override def getMessage: String = s"$message: ${cause.getMessage}"
    override def getCause: Throwable = cause
  
  /**
   * Error that occurs during agent operations.
   *
   * @param cause The underlying cause of the error
   * @param message A descriptive error message
   */
  case class AgentError(
    cause: Throwable,
    message: String = "Agent error occurred"
  ) extends LangChainError:
    override def getMessage: String = s"$message: ${cause.getMessage}"
    override def getCause: Throwable = cause
    
  /**
   * Error that occurs during document loading operations.
   *
   * @param cause The underlying cause of the error
   * @param message A descriptive error message
   */
  case class DocumentLoadingError(
    cause: Throwable,
    message: String = "Document loading error occurred"
  ) extends LangChainError:
    override def getMessage: String = s"$message: ${cause.getMessage}"
    override def getCause: Throwable = cause
    
  /**
   * Error that occurs during document parsing operations.
   *
   * @param cause The underlying cause of the error
   * @param message A descriptive error message
   */
  case class DocumentParsingError(
    cause: Throwable,
    message: String = "Document parsing error occurred"
  ) extends LangChainError:
    override def getMessage: String = s"$message: ${cause.getMessage}"
    override def getCause: Throwable = cause
    
  /**
   * Error that occurs during tool execution.
   *
   * @param cause The underlying cause of the error
   * @param message A descriptive error message
   */
  case class ToolExecutionError(
    cause: Throwable,
    message: String = "Tool execution error occurred"
  ) extends LangChainError:
    override def getMessage: String = s"$message: ${cause.getMessage}"
    override def getCause: Throwable = cause
    
  /**
   * Error that occurs during output parsing operations.
   *
   * @param cause The underlying cause of the error
   * @param message A descriptive error message
   * @param output The original output that failed to parse
   */
  case class OutputParsingError(
    cause: Throwable,
    message: String = "Output parsing error occurred",
    output: Option[String] = None
  ) extends LangChainError:
    override def getMessage: String =
      output match
        case Some(text) => s"$message: ${cause.getMessage}\nOriginal output: $text"
        case None => s"$message: ${cause.getMessage}"
    override def getCause: Throwable = cause