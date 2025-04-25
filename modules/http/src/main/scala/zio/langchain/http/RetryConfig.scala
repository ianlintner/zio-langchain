package zio.langchain.http

import zio.Duration
import zio.durationInt

/**
 * Configuration for HTTP request retries.
 *
 * @param maxRetries Maximum number of retry attempts
 * @param initialDelay Initial delay before the first retry
 * @param backoffFactor Factor by which to increase the delay for each subsequent retry
 * @param retryPredicate Function that determines whether a particular error should trigger a retry
 */
case class RetryConfig(
  maxRetries: Int = 3,
  initialDelay: Duration = 1.second,
  backoffFactor: Double = 2.0,
  retryPredicate: HttpError => Boolean = RetryConfig.defaultRetryPredicate
)

/**
 * Companion object for RetryConfig.
 */
object RetryConfig:
  /**
   * Default retry configuration.
   */
  val default: RetryConfig = RetryConfig()
  
  /**
   * Default predicate for determining whether to retry a request.
   * By default, retries on connection errors and server errors (5xx).
   *
   * @param error The error to check
   * @return True if the request should be retried, false otherwise
   */
  val defaultRetryPredicate: HttpError => Boolean = {
    case _: ConnectionError => true
    case ResponseError(status, _, _) =>
      status.code >= 500 && status.code < 600
    case _ => false
  }