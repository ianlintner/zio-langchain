package zio.langchain.http

import zio.*
import zio.http.*
import zio.json.{JsonDecoder, JsonEncoder}
import zio.stream.ZStream

/**
 * Builder for HTTP requests with a fluent API.
 */
trait RequestBuilder:
  /**
   * Set the URL for the request.
   *
   * @param url The URL to request
   * @return A new RequestBuilder with the URL set
   */
  def url(url: String): RequestBuilder
  
  /**
   * Set the HTTP method for the request.
   *
   * @param method The HTTP method to use
   * @return A new RequestBuilder with the method set
   */
  def method(method: Method): RequestBuilder
  
  /**
   * Add a header to the request.
   *
   * @param name The header name
   * @param value The header value
   * @return A new RequestBuilder with the header added
   */
  def header(name: String, value: String): RequestBuilder
  
  /**
   * Add multiple headers to the request.
   *
   * @param headers The headers to add
   * @return A new RequestBuilder with the headers added
   */
  def headers(headers: Headers): RequestBuilder
  
  /**
   * Add a query parameter to the request.
   *
   * @param name The parameter name
   * @param value The parameter value
   * @return A new RequestBuilder with the query parameter added
   */
  def queryParam(name: String, value: String): RequestBuilder
  
  /**
   * Set the request body.
   *
   * @param body The body content
   * @param encoder The JSON encoder for the body type
   * @return A new RequestBuilder with the body set
   */
  def body[A](body: A)(using encoder: JsonEncoder[A]): RequestBuilder
  
  /**
   * Set the authentication for the request.
   *
   * @param auth The authentication to use
   * @return A new RequestBuilder with the authentication set
   */
  def auth(auth: Auth): RequestBuilder
  
  /**
   * Execute the request and decode the response.
   *
   * @param decoder The JSON decoder for the response type
   * @return A ZIO effect that completes with the decoded response or fails with an HttpError
   */
  def execute[R](using decoder: JsonDecoder[R]): ZIO[Any, zio.langchain.http.HttpError, R]
  
  /**
   * Execute the request and return a stream of decoded elements.
   *
   * @param decoder The JSON decoder for the stream element type
   * @return A ZStream that emits decoded elements or fails with an HttpError
   */
  def executeStream[R](using decoder: JsonDecoder[R]): ZStream[Any, zio.langchain.http.HttpError, R]
  
  /**
   * Build the HTTP request without executing it.
   *
   * @return The built HTTP request
   */
  def build: Request