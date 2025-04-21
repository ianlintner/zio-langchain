package zio.langchain.document.chunkers

import zio.*
import zio.test.*
import zio.test.Assertion.*
import zio.langchain.core.domain.Document

object DocumentChunkerSpec extends ZIOSpecDefault {
  def spec = suite("DocumentChunkerSpec")(
    test("CharacterChunker should split documents by character count") {
      // Given
      val document = Document(
        id = "test-doc",
        content = "This is a test document that will be split into chunks based on character count.",
        metadata = Map("source" -> "test")
      )
      val chunker = DocumentChunkers.byCharacterCount(chunkSize = 20, chunkOverlap = 5)
      
      // When
      val result = chunker.chunk(document)
      
      // Then
      assertZIO(result.map(_.size))(equalTo(4)) &&
      assertZIO(result.map(_.map(_.content)))(
        equalTo(Seq(
          "This is a test docu",
          "ocument that will be",
          "l be split into chun",
          "chunks based on char"
        ))
      )
    },
    
    test("TokenChunker should split documents by estimated token count") {
      // Given
      val document = Document(
        id = "test-doc",
        content = "This is a test document that will be split into chunks based on estimated token count.",
        metadata = Map("source" -> "test")
      )
      val chunker = DocumentChunkers.byTokenCount(chunkSize = 10, chunkOverlap = 2)
      
      // When
      val result = chunker.chunk(document)
      
      // Then
      assertZIO(result.map(_.size))(isGreaterThan(1)) &&
      assertZIO(result.map(_.forall(_.metadata.contains("estimated_tokens"))))(isTrue)
    },
    
    test("SeparatorChunker should split documents by separator") {
      // Given
      val document = Document(
        id = "test-doc",
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        metadata = Map("source" -> "test")
      )
      val chunker = DocumentChunkers.bySeparator(separator = "\n\n")
      
      // When
      val result = chunker.chunk(document)
      
      // Then
      assertZIO(result.map(_.size))(equalTo(3)) &&
      assertZIO(result.map(_.map(_.content)))(
        equalTo(Seq(
          "First paragraph.",
          "Second paragraph.",
          "Third paragraph."
        ))
      )
    },
    
    test("SentenceChunker should split documents by sentences") {
      // Given
      val document = Document(
        id = "test-doc",
        content = "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence.",
        metadata = Map("source" -> "test")
      )
      val chunker = DocumentChunkers.bySentence(maxSentences = 2, overlapSentences = 1)
      
      // When
      val result = chunker.chunk(document)
      
      // Then
      assertZIO(result.map(_.size))(isGreaterThan(1))
    },
    
    test("ParagraphChunker should split documents by paragraphs") {
      // Given
      val document = Document(
        id = "test-doc",
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph.",
        metadata = Map("source" -> "test")
      )
      val chunker = DocumentChunkers.byParagraph(maxParagraphs = 2, overlapParagraphs = 1)
      
      // When
      val result = chunker.chunk(document)
      
      // Then
      assertZIO(result.map(_.size))(equalTo(3)) &&
      assertZIO(result.map(_.map(_.content)))(
        contains(
          "First paragraph.\n\nSecond paragraph."
        ) && 
        contains(
          "Second paragraph.\n\nThird paragraph."
        ) &&
        contains(
          "Third paragraph.\n\nFourth paragraph."
        )
      )
    },
    
    test("RecursiveChunker should apply multiple chunkers in sequence") {
      // Given
      val document = Document(
        id = "test-doc",
        content = "First paragraph with multiple sentences. This is another sentence.\n\nSecond paragraph with more sentences. And another one here.",
        metadata = Map("source" -> "test")
      )
      val paragraphChunker = DocumentChunkers.byParagraph(maxParagraphs = 1)
      val sentenceChunker = DocumentChunkers.bySentence(maxSentences = 1)
      val recursiveChunker = DocumentChunkers.recursive(paragraphChunker, sentenceChunker)
      
      // When
      val result = recursiveChunker.chunk(document)
      
      // Then
      assertZIO(result.map(_.size))(equalTo(4)) &&
      assertZIO(result.map(_.forall(_.metadata.contains("recursive_level"))))(isTrue)
    },
    
    test("SemanticChunker should add semantic metadata") {
      // Given
      val document = Document(
        id = "test-doc",
        content = "This is a test document for semantic chunking.",
        metadata = Map("source" -> "test")
      )
      val initialChunker = DocumentChunkers.byCharacterCount(chunkSize = 20)
      val semanticChunker = DocumentChunkers.semantic(similarityThreshold = 0.8, initialChunker)
      
      // When
      val result = semanticChunker.chunk(document)
      
      // Then
      assertZIO(result.map(_.forall(_.metadata.contains("similarity_threshold"))))(isTrue)
    },
    
    test("Chunkers can be composed with andThen") {
      // Given
      val document = Document(
        id = "test-doc",
        content = "First paragraph.\n\nSecond paragraph.",
        metadata = Map("source" -> "test")
      )
      val paragraphChunker = DocumentChunkers.byParagraph(maxParagraphs = 1)
      val characterChunker = DocumentChunkers.byCharacterCount(chunkSize = 10, chunkOverlap = 0)
      val composedChunker = paragraphChunker >>> characterChunker
      
      // When
      val result = composedChunker.chunk(document)
      
      // Then
      assertZIO(result.map(_.size))(isGreaterThan(2))
    }
  )
}