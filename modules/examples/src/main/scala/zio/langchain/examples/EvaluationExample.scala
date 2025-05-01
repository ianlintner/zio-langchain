package zio.langchain.examples

import zio.*
import zio.http.Client

import zio.langchain.core.domain.*
import zio.langchain.core.model.LLM
import zio.langchain.evaluation.*
import zio.langchain.integrations.openai.*

/**
 * Example demonstrating the use of evaluation metrics for RAG systems and text generation.
 *
 * To run this example:
 * 1. Set your OpenAI API key in the environment variable OPENAI_API_KEY
 * 2. Run the example using: `sbt "examples/runMain zio.langchain.examples.EvaluationExample"`
 */
object EvaluationExample extends ZIOAppDefault {

  // Sample documents with the fixed id parameter using String directly (not Some[String])
  val sampleDocuments = Seq(
    Document(
      id = "doc1", // Fixed: Changed from Some("doc1") to "doc1"
      content = "ZIO is a library for asynchronous and concurrent programming in Scala.",
      metadata = Map("source" -> "zio-docs")
    ),
    Document(
      id = "doc2", // Fixed: Changed from Some("doc2") to "doc2"
      content = "ZIO offers a robust error handling model with typed errors.",
      metadata = Map("source" -> "zio-examples")
    )
  )

  override def run = ZIO.succeed(()).exitCode
}