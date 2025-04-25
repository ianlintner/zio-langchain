package zio.langchain.http

import zio.http.Headers

/**
 * Authentication mechanism for HTTP requests.
 */
trait Auth:
  /**
   * Apply authentication to the given headers.
   *
   * @param headers The headers to apply authentication to
   * @return The headers with authentication applied
   */
  def applyToHeaders(headers: Headers): Headers

/**
 * Bearer token authentication.
 *
 * @param token The bearer token
 */
case class BearerTokenAuth(token: String) extends Auth:
  override def applyToHeaders(headers: Headers): Headers =
    headers ++ Headers("Authorization" -> s"Bearer $token")

/**
 * API key authentication.
 *
 * @param key The API key
 * @param headerName The name of the header to use for the API key (default: "X-API-Key")
 */
case class ApiKeyAuth(key: String, headerName: String = "X-API-Key") extends Auth:
  override def applyToHeaders(headers: Headers): Headers =
    headers ++ Headers(headerName -> key)

/**
 * Basic authentication.
 *
 * @param username The username
 * @param password The password
 */
case class BasicAuth(username: String, password: String) extends Auth:
  import java.util.Base64
  
  override def applyToHeaders(headers: Headers): Headers =
    val credentials = Base64.getEncoder.encodeToString(s"$username:$password".getBytes)
    headers ++ Headers("Authorization" -> s"Basic $credentials")

/**
 * Custom header authentication.
 *
 * @param headers The custom headers to add
 */
case class CustomHeaderAuth(customHeaders: Map[String, String]) extends Auth:
  override def applyToHeaders(headers: Headers): Headers =
    customHeaders.foldLeft(headers) { case (acc, (name, value)) =>
      acc ++ Headers(name -> value)
    }