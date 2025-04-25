package zio.langchain.http

import zio.http.Status

/**
 * Base trait for HTTP errors.
 */
sealed trait HttpError extends Throwable:
  /**
   * Error message.
   */
  def message: String
  
  /**
   * Optional cause of the error.
   */
  def cause: Option[Throwable]
  
  override def getMessage: String = message
  
  override def getCause: Throwable = cause.orNull

/**
 * Error that occurs when building or sending a request.
 *
 * @param message Error message
 * @param cause Optional cause of the error
 */
case class RequestError(
  message: String,
  cause: Option[Throwable] = None
) extends HttpError

/**
 * Error that occurs when receiving an unsuccessful response.
 *
 * @param status HTTP status code
 * @param message Error message
 * @param body Optional response body
 */
case class ResponseError(
  status: Status,
  message: String,
  body: Option[String] = None
) extends HttpError:
  override val cause: Option[Throwable] = None

/**
 * Error that occurs when a request times out.
 *
 * @param message Error message
 * @param cause Optional cause of the error
 */
case class TimeoutError(
  message: String,
  cause: Option[Throwable] = None
) extends HttpError

/**
 * Error that occurs when serializing or deserializing data.
 *
 * @param message Error message
 * @param cause Optional cause of the error
 */
case class SerializationError(
  message: String,
  cause: Option[Throwable] = None
) extends HttpError

/**
 * Error that occurs when connecting to a server.
 *
 * @param message Error message
 * @param cause Optional cause of the error
 */
case class ConnectionError(
  message: String,
  cause: Option[Throwable] = None
) extends HttpError