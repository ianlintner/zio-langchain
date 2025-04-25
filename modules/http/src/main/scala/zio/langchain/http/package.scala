package zio.langchain

import zio.*
import zio.http.{Client, Request, Response, Headers, Method, Body, Status}
import zio.stream.ZStream

/**
 * HTTP client module for ZIO-Langchain.
 *
 * This module provides a type-safe, ZIO-based HTTP client with support for:
 * - JSON serialization/deserialization
 * - Streaming responses
 * - Retry policies
 * - Authentication mechanisms
 * - Error handling
 *
 * Example usage:
 * {{{
 * import zio.*
 * import zio.langchain.http.*
 * import zio.json.*
 *
 * case class User(id: Int, name: String)
 * object User {
 *   implicit val decoder: JsonDecoder[User] = DeriveJsonDecoder.gen[User]
 * }
 *
 * val program = for {
 *   client <- HttpClient.make()
 *   user <- client.get[User]("https://api.example.com/users/1")
 *   _ <- Console.printLine(s"User: ${user.name}")
 * } yield ()
 *
 * val run = program.provide(Client.default)
 * }}}
 */
package object http:
  /**
   * Type alias for HTTP client effects.
   */
  type HttpClientEffect[R] = ZIO[Client, Nothing, HttpClient]
  
  /**
   * Type alias for HTTP request effects.
   */
  type HttpRequestEffect[R] = ZIO[Any, HttpError, R]
  
  /**
   * Type alias for HTTP streaming effects.
   */
  type HttpStreamEffect[R] = ZStream[Any, HttpError, R]