package zio.langchain.http

import zio.*
import zio.http.*
import zio.json.{JsonDecoder, JsonEncoder}
import zio.stream.ZStream
import zio.langchain.http.streaming.HttpStreaming.JsonStreamingPipeline

/**
 * Client for making HTTP requests with support for JSON serialization/deserialization,
 * error handling, and retries.
 */
trait HttpClient:
  /**
   * Performs a GET request.
   *
   * @param url The URL to request
   * @param headers Optional headers to include
   * @param decoder JSON decoder for the response type
   * @return ZIO effect that completes with the decoded response or fails with an HttpError
   */
  def get[A](url: String, headers: Headers = Headers.empty)
    (using decoder: JsonDecoder[A]): ZIO[Any, zio.langchain.http.HttpError, A]
  
  /**
   * Performs a POST request.
   *
   * @param url The URL to request
   * @param body The request body
   * @param headers Optional headers to include
   * @param encoder JSON encoder for the request body type
   * @param decoder JSON decoder for the response type
   * @return ZIO effect that completes with the decoded response or fails with an HttpError
   */
  def post[A, B](url: String, body: A, headers: Headers = Headers.empty)
    (using encoder: JsonEncoder[A], decoder: JsonDecoder[B]): ZIO[Any, zio.langchain.http.HttpError, B]
  
  /**
   * Performs a PUT request.
   *
   * @param url The URL to request
   * @param body The request body
   * @param headers Optional headers to include
   * @param encoder JSON encoder for the request body type
   * @param decoder JSON decoder for the response type
   * @return ZIO effect that completes with the decoded response or fails with an HttpError
   */
  def put[A, B](url: String, body: A, headers: Headers = Headers.empty)
    (using encoder: JsonEncoder[A], decoder: JsonDecoder[B]): ZIO[Any, zio.langchain.http.HttpError, B]
  
  /**
   * Performs a DELETE request.
   *
   * @param url The URL to request
   * @param headers Optional headers to include
   * @param decoder JSON decoder for the response type
   * @return ZIO effect that completes with the decoded response or fails with an HttpError
   */
  def delete[A](url: String, headers: Headers = Headers.empty)
    (using decoder: JsonDecoder[A]): ZIO[Any, zio.langchain.http.HttpError, A]
  
  /**
   * Performs a PATCH request.
   *
   * @param url The URL to request
   * @param body The request body
   * @param headers Optional headers to include
   * @param encoder JSON encoder for the request body type
   * @param decoder JSON decoder for the response type
   * @return ZIO effect that completes with the decoded response or fails with an HttpError
   */
  def patch[A, B](url: String, body: A, headers: Headers = Headers.empty)
    (using encoder: JsonEncoder[A], decoder: JsonDecoder[B]): ZIO[Any, zio.langchain.http.HttpError, B]
  
  /**
   * Performs a HEAD request.
   *
   * @param url The URL to request
   * @param headers Optional headers to include
   * @return ZIO effect that completes with the response headers or fails with an HttpError
   */
  def head(url: String, headers: Headers = Headers.empty): ZIO[Any, zio.langchain.http.HttpError, Headers]
  
  /**
   * Returns a request builder for constructing custom requests.
   *
   * @return A new RequestBuilder instance
   */
  def request: RequestBuilder
  
  /**
   * Executes a streaming request.
   *
   * @param request The HTTP request to execute
   * @param decoder JSON decoder for the stream elements
   * @return ZStream that emits decoded elements or fails with an HttpError
   */
  def stream[A](request: Request)(using decoder: JsonDecoder[A]): ZStream[Any, zio.langchain.http.HttpError, A]

/**
 * Companion object for HttpClient.
 */
object HttpClient:
  /**
   * Creates a new HttpClient with the specified configuration.
   *
   * @param config The client configuration
   * @param jsonCodecProvider The JSON codec provider to use
   * @return A new HttpClient instance
   */
  def make(
    config: HttpClientConfig = HttpClientConfig.default,
    jsonCodecProvider: JsonCodecProvider = ZioJsonProvider
  ): ZIO[Client, Nothing, HttpClient] = 
    for
      client <- ZIO.service[Client]
    yield new HttpClientLive(client, config, jsonCodecProvider)

/**
 * Live implementation of HttpClient.
 */
private class HttpClientLive(
  client: Client,
  config: HttpClientConfig,
  jsonCodecProvider: JsonCodecProvider
) extends HttpClient:
  
  override def get[A](url: String, headers: Headers)
    (using decoder: JsonDecoder[A]): ZIO[Any, zio.langchain.http.HttpError, A] =
    request.url(url).method(Method.GET).headers(headers).execute[A]
  
  override def post[A, B](url: String, body: A, headers: Headers)
    (using encoder: JsonEncoder[A], decoder: JsonDecoder[B]): ZIO[Any, zio.langchain.http.HttpError, B] =
    request.url(url).method(Method.POST).headers(headers).body(body).execute[B]
  
  override def put[A, B](url: String, body: A, headers: Headers)
    (using encoder: JsonEncoder[A], decoder: JsonDecoder[B]): ZIO[Any, zio.langchain.http.HttpError, B] =
    request.url(url).method(Method.PUT).headers(headers).body(body).execute[B]
  
  override def delete[A](url: String, headers: Headers)
    (using decoder: JsonDecoder[A]): ZIO[Any, zio.langchain.http.HttpError, A] =
    request.url(url).method(Method.DELETE).headers(headers).execute[A]
  
  override def patch[A, B](url: String, body: A, headers: Headers)
    (using encoder: JsonEncoder[A], decoder: JsonDecoder[B]): ZIO[Any, zio.langchain.http.HttpError, B] =
    request.url(url).method(Method.PATCH).headers(headers).body(body).execute[B]
  
  override def head(url: String, headers: Headers): ZIO[Any, zio.langchain.http.HttpError, Headers] =
    client.request(request.url(url).method(Method.HEAD).headers(headers).build)
      .mapError(e => zio.langchain.http.ConnectionError(s"Failed to send HEAD request: ${e.getMessage}", Some(e)))
      .flatMap { response =>
        if (response.status.isSuccess)
          ZIO.succeed(response.headers)
        else
          ZIO.fail(zio.langchain.http.ResponseError(
            response.status,
            s"HEAD request failed with status ${response.status.code}",
            None
          ))
      }
  
  override def request: RequestBuilder = new RequestBuilderLive(client, config, jsonCodecProvider)
  
  override def stream[A](request: Request)(using decoder: JsonDecoder[A]): ZStream[Any, zio.langchain.http.HttpError, A] =
    val sendRequest = client.request(request)
      .mapError(e => zio.langchain.http.ConnectionError(s"Failed to send streaming request: ${e.getMessage}", Some(e)))
    
    val execution = sendRequest.flatMap { response =>
      if (response.status.isSuccess)
        ZIO.succeed(response)
      else
        response.body.asString.flatMap { body =>
          ZIO.fail(zio.langchain.http.ResponseError(
            response.status,
            s"Streaming request failed with status ${response.status.code}",
            Some(body)
          ))
        }
    }
    
    // Handle retries if configured
    val result = config.retryConfig match
      case Some(retryConfig) =>
        execution.retry(
          Schedule.recurs(retryConfig.maxRetries) &&
          Schedule.exponential(retryConfig.initialDelay, retryConfig.backoffFactor) &&
          Schedule.recurWhile[Throwable](e =>
            e match {
              case httpError: zio.langchain.http.HttpError => retryConfig.retryPredicate(httpError)
              case _ => false
            }
          )
        )
      case None => execution
    
    // Ensure all errors are of type zio.langchain.http.HttpError
    val responseEffect = result.mapError {
      case e: zio.langchain.http.HttpError => e
      case e => zio.langchain.http.ConnectionError(s"Unexpected error: ${e.getMessage}", Some(e))
    }
    
    ZStream.fromZIO(responseEffect).flatMap { response =>
      response.body.asStream
        .mapError(err => zio.langchain.http.SerializationError(s"Stream error: ${err.getMessage}", Some(err)))
        .via(JsonStreamingPipeline.jsonLines[A](jsonCodecProvider))
    }

/**
 * Live implementation of RequestBuilder.
 */
private class RequestBuilderLive(
  client: Client,
  config: HttpClientConfig,
  jsonCodecProvider: JsonCodecProvider,
  private val currentUrl: Option[String] = None,
  private val currentMethod: Method = Method.GET,
  private val currentHeaders: Headers = Headers.empty,
  private val currentQueryParams: Map[String, String] = Map.empty,
  private val currentBody: Option[Body] = None,
  private val currentAuth: Option[Auth] = None
) extends RequestBuilder:
  
  override def url(url: String): RequestBuilder =
    new RequestBuilderLive(client, config, jsonCodecProvider, Some(url), currentMethod, 
      currentHeaders, currentQueryParams, currentBody, currentAuth)
  
  override def method(method: Method): RequestBuilder =
    new RequestBuilderLive(client, config, jsonCodecProvider, currentUrl, method, 
      currentHeaders, currentQueryParams, currentBody, currentAuth)
  
  override def header(name: String, value: String): RequestBuilder =
    new RequestBuilderLive(client, config, jsonCodecProvider, currentUrl, currentMethod, 
      currentHeaders ++ Headers(Header.Custom(name, value)), currentQueryParams, currentBody, currentAuth)
  
  override def headers(headers: Headers): RequestBuilder =
    new RequestBuilderLive(client, config, jsonCodecProvider, currentUrl, currentMethod, 
      currentHeaders ++ headers, currentQueryParams, currentBody, currentAuth)
  
  override def queryParam(name: String, value: String): RequestBuilder =
    new RequestBuilderLive(client, config, jsonCodecProvider, currentUrl, currentMethod, 
      currentHeaders, currentQueryParams + (name -> value), currentBody, currentAuth)
  
  override def body[A](body: A)(using encoder: JsonEncoder[A]): RequestBuilder =
    val jsonBody = jsonCodecProvider.encode(body)
    new RequestBuilderLive(client, config, jsonCodecProvider, currentUrl, currentMethod, 
      currentHeaders, currentQueryParams, Some(Body.fromString(jsonBody)), currentAuth)
  
  override def auth(auth: Auth): RequestBuilder =
    new RequestBuilderLive(client, config, jsonCodecProvider, currentUrl, currentMethod, 
      currentHeaders, currentQueryParams, currentBody, Some(auth))
  
  override def execute[R](using decoder: JsonDecoder[R]): ZIO[Any, zio.langchain.http.HttpError, R] = {
    for {
      req <- ZIO.fromEither(buildRequest).mapError(msg => RequestError(msg))
      resp <- client.request(req).mapError(e => ConnectionError(s"Failed to send request: ${e.getMessage}", Some(e)))
      result <-
        if (resp.status.isSuccess) {
          for {
            body <- resp.body.asString.mapError(e => ConnectionError(s"Failed to read response body: ${e.getMessage}", Some(e)))
            decoded <- ZIO.fromEither(jsonCodecProvider.decode[R](body))
              .mapError(msg => SerializationError(s"Failed to decode response: $msg"))
          } yield decoded
        } else {
          for {
            body <- resp.body.asString.mapError(e => ConnectionError(s"Failed to read response body: ${e.getMessage}", Some(e)))
            result <- ZIO.fail(ResponseError(
              resp.status,
              s"Request failed with status ${resp.status.code}",
              Some(body)
            ))
          } yield result
        }
    } yield result
  }
  
  override def executeStream[R](using decoder: JsonDecoder[R]): ZStream[Any, zio.langchain.http.HttpError, R] = {
    for {
      req <- ZStream.fromZIO(
        ZIO.fromEither(buildRequest).mapError(msg => RequestError(msg))
      )
      resp <- ZStream.fromZIO(
        client.request(req).mapError(e => ConnectionError(s"Failed to send request: ${e.getMessage}", Some(e)))
      )
      result <-
        if (resp.status.isSuccess) {
          // Success case - convert the response body to a stream
          resp.body.asStream
            .mapError(err => SerializationError(s"Stream error: ${err.getMessage}", Some(err)))
            .via(JsonStreamingPipeline.jsonLines[R](jsonCodecProvider))
        } else {
          // Error case - read the body and create a failing stream
          ZStream.fromZIO(
            resp.body.asString
              .mapError(e => ConnectionError(s"Failed to read response body: ${e.getMessage}", Some(e)))
              .flatMap { body =>
                ZIO.fail(ResponseError(
                  resp.status,
                  s"Request failed with status ${resp.status.code}",
                  Some(body)
                ))
              }
          )
        }
    } yield result
  }
  
  override def build: Request =
    buildRequest.getOrElse(throw new IllegalStateException("Failed to build request"))
  
  private def buildRequest: Either[String, Request] =
    for
      url <- currentUrl.toRight("URL is required")
      urlWithParams = if (currentQueryParams.isEmpty) url else
        url + "?" + currentQueryParams.map { case (k, v) => s"$k=$v" }.mkString("&")
      request = Request(
        url = URL.decode(urlWithParams).toOption.getOrElse(throw new IllegalArgumentException(s"Invalid URL: $urlWithParams")),
        method = currentMethod,
        headers = currentAuth.fold(currentHeaders)(auth => auth.applyToHeaders(currentHeaders)),
        body = currentBody.getOrElse(Body.empty),
        version = Version.Http_1_1,
        remoteAddress = None
      )
    yield request
  