package zio.langchain

import zio.langchain.core.errors.LangChainError

/**
 * Evaluation module for ZIO LangChain.
 * Provides tools and utilities for evaluating LLM and RAG performance.
 */
package object evaluation:
  /**
   * Error that occurs during evaluation operations.
   *
   * @param cause The underlying cause of the error
   * @param message A descriptive error message
   */
  case class EvaluationError(
    cause: Throwable,
    message: String = "Evaluation error occurred"
  ) extends LangChainError:
    override def getMessage: String = s"$message: ${cause.getMessage}"
    override def getCause: Throwable = cause