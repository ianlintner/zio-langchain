package zio.langchain.http

import zio.Duration
import zio.durationInt

/**
 * Configuration for the HTTP client.
 *
 * @param connectTimeout Timeout for establishing a connection
 * @param readTimeout Timeout for reading data from an established connection
 * @param retryConfig Optional retry configuration (if None, no retries will be performed)
 * @param followRedirects Whether to automatically follow redirects
 * @param maxRedirects Maximum number of redirects to follow
 */
case class HttpClientConfig(
  connectTimeout: Duration = 30.seconds,
  readTimeout: Duration = 60.seconds,
  retryConfig: Option[RetryConfig] = Some(RetryConfig.default),
  followRedirects: Boolean = true,
  maxRedirects: Int = 5
)

/**
 * Companion object for HttpClientConfig.
 */
object HttpClientConfig:
  /**
   * Default HTTP client configuration.
   */
  val default: HttpClientConfig = HttpClientConfig()
  
  /**
   * Configuration with no retries.
   */
  val noRetries: HttpClientConfig = HttpClientConfig(retryConfig = None)
  
  /**
   * Configuration with short timeouts for quick operations.
   */
  val shortTimeouts: HttpClientConfig = HttpClientConfig(
    connectTimeout = 5.seconds,
    readTimeout = 10.seconds
  )
  
  /**
   * Configuration with long timeouts for operations that may take a long time.
   */
  val longTimeouts: HttpClientConfig = HttpClientConfig(
    connectTimeout = 60.seconds,
    readTimeout = 300.seconds
  )