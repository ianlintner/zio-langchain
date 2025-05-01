package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client

import zio.langchain.core.domain.Document
import zio.langchain.document.chunkers.{DocumentChunker, DocumentChunkers}

/**
 * A demonstration of document chunking functionality using various chunking strategies.
 * 
 * This example shows how to:
 * 1. Create documents from text
 * 2. Apply different chunking strategies (character, token, paragraph, etc.)
 * 3. Compose chunkers together
 * 4. Handle chunking errors properly
 */
object DocumentChunkingExample extends ZIOAppDefault:

  override def run =
    program.catchAllCause { cause =>
      val error = cause.failureOption.getOrElse(new RuntimeException("Unknown error"))
      for {
        _ <- printLine(s"Error: ${error.getMessage}").orDie
      } yield ExitCode.failure
    }

  val program = for
    _ <- printLine("Welcome to ZIO LangChain Document Chunking Example!")
    _ <- printLine("This example demonstrates different document chunking strategies.")
    _ <- printLine("")

    // Create example documents
    document = createExampleDocument()
    longDocument = createLongExampleDocument()

    // PART 1: Basic Chunking Strategies
    _ <- printLine("=== PART 1: Basic Chunking Strategies ===")
    
    // Example 1: Character chunking
    _ <- printLine("1. Character chunking (chunk size: 50, overlap: 10):")
    characterChunker = DocumentChunkers.byCharacterCount(chunkSize = 50, chunkOverlap = 10)
    charChunks <- characterChunker.chunk(document)
    _ <- printChunks(charChunks)
    
    // Example 2: Token chunking (estimated)
    _ <- printLine("\n2. Token chunking (chunk size: 20 tokens, overlap: 5 tokens):")
    tokenChunker = DocumentChunkers.byTokenCount(chunkSize = 20, chunkOverlap = 5)
    tokenChunks <- tokenChunker.chunk(document)
    _ <- printChunks(tokenChunks)
    
    // Example 3: Separator chunking
    _ <- printLine("\n3. Separator chunking (separator: '.'):")
    separatorChunker = DocumentChunkers.bySeparator(separator = ".", keepSeparator = true)
    separatorChunks <- separatorChunker.chunk(document)
    _ <- printChunks(separatorChunks)

    // PART 2: Advanced Chunking Strategies
    _ <- printLine("\n=== PART 2: Advanced Chunking Strategies ===")
    
    // Example 4: Paragraph chunking
    _ <- printLine("4. Paragraph chunking (max: 2 paragraphs, overlap: 1):")
    paragraphChunker = DocumentChunkers.byParagraph(maxParagraphs = 2, overlapParagraphs = 1)
    paragraphChunks <- paragraphChunker.chunk(longDocument)
    _ <- printChunks(paragraphChunks)
    
    // Example 5: Sentence chunking
    _ <- printLine("\n5. Sentence chunking (max: 2 sentences, overlap: 1):")
    sentenceChunker = DocumentChunkers.bySentence(maxSentences = 2, overlapSentences = 1)
    sentenceChunks <- sentenceChunker.chunk(longDocument)
    _ <- printChunks(sentenceChunks)

    // PART 3: Composing Chunkers
    _ <- printLine("\n=== PART 3: Composing Chunkers ===")
    
    // Example 6: Composing chunkers (paragraph then sentence)
    _ <- printLine("6. Composing chunkers (paragraph then token):")
    composedChunker = DocumentChunkers.byParagraph(maxParagraphs = 1) >>>
                     DocumentChunkers.byTokenCount(chunkSize = 15, chunkOverlap = 0)
    composedChunks <- composedChunker.chunk(longDocument)
    _ <- printChunks(composedChunks)

    // PART 4: Batch Processing
    _ <- printLine("\n=== PART 4: Batch Processing ===")
    
    // Example 7: Processing multiple documents
    _ <- printLine("7. Processing multiple documents:")
    documents = Seq(
      Document(id = "doc1", content = "This is the first test document for batch processing."),
      Document(id = "doc2", content = "This is the second test document for batch processing.")
    )
    batchChunker = DocumentChunkers.byCharacterCount(chunkSize = 20, chunkOverlap = 5)
    batchChunks <- batchChunker.chunkAll(documents)
    _ <- printChunks(batchChunks)

    // Final message
    _ <- printLine("\nAll chunking examples completed successfully!")
  yield ()

  /**
   * Creates a simple example document for chunking demonstrations.
   */
  private def createExampleDocument(): Document =
    Document(
      id = "example-doc-1",
      content = "This is a test document that will be split into chunks using different strategies. " +
                "Each strategy has different parameters and produces different results. " +
                "Chunking is an essential part of processing documents for retrieval.",
      metadata = Map("source" -> "example", "category" -> "test")
    )

  /**
   * Creates a longer example document with multiple paragraphs for advanced chunking demonstrations.
   */
  private def createLongExampleDocument(): Document =
    Document(
      id = "example-doc-2",
      content = 
        """First paragraph with multiple sentences. This is the second sentence. Here is the third sentence.

        |Second paragraph that also has several sentences. Another sentence here. And one final sentence.

        |Third paragraph with some content. More text goes here.

        |Fourth paragraph to complete our example document. The final sentence.""".stripMargin,
      metadata = Map("source" -> "example", "category" -> "test")
    )

  /**
   * Helper function to print chunks in a formatted way.
   */
  private def printChunks(chunks: Seq[Document]): ZIO[Any, java.io.IOException, Unit] =
    ZIO.foreach(chunks.zipWithIndex) { case (chunk, i) =>
      printLine(s"  Chunk ${i+1}:")
        .zipRight(printLine(s"  - Content: \"${chunk.content.replace("\n", "\\n")}\""))
        .zipRight(printLine(s"  - Metadata: ${chunk.metadata.mkString(", ")}"))
    }.unit