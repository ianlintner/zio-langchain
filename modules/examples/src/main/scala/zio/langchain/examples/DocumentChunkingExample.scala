package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.stream.ZStream

import zio.langchain.core.model.*
import zio.langchain.core.retriever.*
import zio.langchain.core.document.*
import zio.langchain.core.domain.*
import zio.langchain.core.chain.*
import zio.langchain.integrations.openai.*
import zio.langchain.core.errors.{LangChainError, RetrieverError, LLMError}
import zio.langchain.document.chunkers.*

import java.nio.file.{Path, Files, Paths}
import scala.jdk.CollectionConverters.*

/**
 * An example demonstrating different document chunking strategies for RAG.
 * This example shows how to:
 * 1. Load a document from a text file
 * 2. Apply different chunking strategies
 * 3. Compare the results of different chunking approaches
 * 4. Use the chunks in a RAG system
 */
object DocumentChunkingExample extends ZIOAppDefault:
  /**
   * The main program.
   */
  override def run: ZIO[Any, Throwable, Unit] =
    // Create the program
    val program = for
      // Print welcome message
      _ <- ZIO.logInfo("Welcome to ZIO LangChain Document Chunking Example!")
      
      // Get the services
      llm <- ZIO.service[LLM]
      embeddingModel <- ZIO.service[EmbeddingModel]
      
      // Load a sample document
      document <- loadSampleDocument
      _ <- ZIO.logInfo(s"Loaded document with ${document.content.length} characters")
      
      // Demonstrate different chunking strategies
      _ <- demonstrateChunkingStrategies(document)
      
      // Create a RAG system with different chunking strategies
      _ <- compareRAGWithDifferentChunkers(document, llm, embeddingModel)
    yield ()
    
    // Provide the required services and run the program
    program.provide(
      // OpenAI LLM layer
      OpenAILLM.live,
      // OpenAI LLM configuration layer
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "gpt-3.5-turbo",
          temperature = 0.0 // Use a low temperature for factual responses
        )
      ),
      // OpenAI Embedding layer
      OpenAIEmbedding.live,
      // OpenAI Embedding configuration layer
      ZLayer.succeed(
        OpenAIEmbeddingConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "text-embedding-ada-002"
        )
      )
    )
  
  /**
   * Loads a sample document from a file or creates one if the file doesn't exist.
   *
   * @return A ZIO effect that produces a document
   */
  private def loadSampleDocument: ZIO[Any, Throwable, Document] =
    val samplePath = Paths.get("sample-document.txt")
    
    for
      exists <- ZIO.attempt(Files.exists(samplePath))
      document <- if exists then
        for
          content <- ZIO.attempt(Files.readString(samplePath))
          document = Document(
            id = "sample-doc",
            content = content,
            metadata = Map("source" -> "sample-document.txt")
          )
        yield document
      else
        // Create a sample document if it doesn't exist
        val sampleContent = """
          |# ZIO LangChain: A Scala Framework for LLM Applications
          |
          |## Introduction
          |
          |ZIO LangChain is a Scala framework for building applications powered by large language models (LLMs). It provides a comprehensive set of tools and abstractions for working with LLMs, document processing, retrieval, and more.
          |
          |## Core Concepts
          |
          |### Documents
          |
          |Documents are the basic unit of data in ZIO LangChain. A document consists of content (text) and metadata. Documents can be loaded from various sources, parsed, and processed in different ways.
          |
          |### Embeddings
          |
          |Embeddings are vector representations of text that capture semantic meaning. ZIO LangChain provides abstractions for working with embedding models and using embeddings for retrieval.
          |
          |### Retrievers
          |
          |Retrievers are responsible for finding relevant documents based on a query. They typically use embeddings to measure semantic similarity between the query and documents.
          |
          |### Chains
          |
          |Chains combine multiple components into a single pipeline. For example, a RAG chain might combine a retriever, a prompt template, and an LLM to answer questions based on retrieved documents.
          |
          |### Agents
          |
          |Agents are autonomous systems that can use tools and make decisions based on user input. They typically use an LLM as the reasoning engine and have access to various tools.
          |
          |## Document Processing
          |
          |Document processing is a critical part of many LLM applications, especially retrieval-augmented generation (RAG) systems. ZIO LangChain provides various tools for document loading, parsing, and chunking.
          |
          |### Document Chunking
          |
          |Document chunking is the process of splitting documents into smaller pieces that can be processed by LLMs and vector databases. Different chunking strategies can significantly impact the performance of RAG systems.
          |
          |#### Character-based Chunking
          |
          |Character-based chunking splits documents based on a fixed number of characters. This is a simple approach but may not respect semantic boundaries.
          |
          |#### Token-based Chunking
          |
          |Token-based chunking splits documents based on a fixed number of tokens. This is more relevant for LLMs, which process text as tokens.
          |
          |#### Sentence-based Chunking
          |
          |Sentence-based chunking splits documents at sentence boundaries. This preserves the semantic coherence of sentences.
          |
          |#### Paragraph-based Chunking
          |
          |Paragraph-based chunking splits documents at paragraph boundaries. This preserves the semantic coherence of paragraphs.
          |
          |#### Recursive Chunking
          |
          |Recursive chunking applies multiple chunking strategies in sequence. For example, first splitting by paragraphs, then by sentences.
          |
          |#### Semantic Chunking
          |
          |Semantic chunking splits documents based on semantic meaning, ensuring that related content stays together.
          |
          |## Conclusion
          |
          |ZIO LangChain provides a powerful set of tools for building LLM applications in Scala. By leveraging ZIO's functional programming model, it offers a type-safe, testable, and composable approach to working with LLMs.
          |""".stripMargin
        
        for
          _ <- ZIO.attempt(Files.writeString(samplePath, sampleContent))
          document = Document(
            id = "sample-doc",
            content = sampleContent,
            metadata = Map("source" -> "sample-document.txt")
          )
        yield document
    yield document
  
  /**
   * Demonstrates different chunking strategies on a document.
   *
   * @param document The document to chunk
   * @return A ZIO effect that completes when the demonstration is done
   */
  private def demonstrateChunkingStrategies(document: Document): ZIO[Any, Throwable, Unit] =
    for
      _ <- ZIO.logInfo("\n=== Demonstrating Different Chunking Strategies ===")
      
      // Character-based chunking
      _ <- ZIO.logInfo("\n--- Character-based Chunking ---")
      characterChunker = DocumentChunkers.byCharacterCount(chunkSize = 200, chunkOverlap = 50)
      characterChunks <- characterChunker.chunk(document)
      _ <- ZIO.logInfo(s"Created ${characterChunks.size} character-based chunks")
      _ <- ZIO.logInfo(s"First chunk: ${characterChunks.headOption.map(_.content.take(100)).getOrElse("")}...")
      
      // Token-based chunking
      _ <- ZIO.logInfo("\n--- Token-based Chunking ---")
      tokenChunker = DocumentChunkers.byTokenCount(chunkSize = 50, chunkOverlap = 10)
      tokenChunks <- tokenChunker.chunk(document)
      _ <- ZIO.logInfo(s"Created ${tokenChunks.size} token-based chunks")
      _ <- ZIO.logInfo(s"First chunk: ${tokenChunks.headOption.map(_.content.take(100)).getOrElse("")}...")
      
      // Sentence-based chunking
      _ <- ZIO.logInfo("\n--- Sentence-based Chunking ---")
      sentenceChunker = DocumentChunkers.bySentence(maxSentences = 5, overlapSentences = 1)
      sentenceChunks <- sentenceChunker.chunk(document)
      _ <- ZIO.logInfo(s"Created ${sentenceChunks.size} sentence-based chunks")
      _ <- ZIO.logInfo(s"First chunk: ${sentenceChunks.headOption.map(_.content.take(100)).getOrElse("")}...")
      
      // Paragraph-based chunking
      _ <- ZIO.logInfo("\n--- Paragraph-based Chunking ---")
      paragraphChunker = DocumentChunkers.byParagraph(maxParagraphs = 2, overlapParagraphs = 1)
      paragraphChunks <- paragraphChunker.chunk(document)
      _ <- ZIO.logInfo(s"Created ${paragraphChunks.size} paragraph-based chunks")
      _ <- ZIO.logInfo(s"First chunk: ${paragraphChunks.headOption.map(_.content.take(100)).getOrElse("")}...")
      
      // Recursive chunking
      _ <- ZIO.logInfo("\n--- Recursive Chunking ---")
      recursiveChunker = DocumentChunkers.recursive(
        DocumentChunkers.byParagraph(maxParagraphs = 3),
        DocumentChunkers.bySentence(maxSentences = 2)
      )
      recursiveChunks <- recursiveChunker.chunk(document)
      _ <- ZIO.logInfo(s"Created ${recursiveChunks.size} recursive chunks")
      _ <- ZIO.logInfo(s"First chunk: ${recursiveChunks.headOption.map(_.content.take(100)).getOrElse("")}...")
    yield ()
  
  /**
   * Compares RAG performance with different chunking strategies.
   *
   * @param document The document to use
   * @param llm The LLM service
   * @param embeddingModel The embedding model service
   * @return A ZIO effect that completes when the comparison is done
   */
  private def compareRAGWithDifferentChunkers(
    document: Document,
    llm: LLM,
    embeddingModel: EmbeddingModel
  ): ZIO[Any, Throwable, Unit] =
    for
      _ <- ZIO.logInfo("\n=== Comparing RAG with Different Chunking Strategies ===")
      
      // Define chunking strategies to compare
      chunkers = Map(
        "Character-based" -> DocumentChunkers.byCharacterCount(chunkSize = 200, chunkOverlap = 50),
        "Paragraph-based" -> DocumentChunkers.byParagraph(maxParagraphs = 2, overlapParagraphs = 1),
        "Sentence-based" -> DocumentChunkers.bySentence(maxSentences = 5, overlapSentences = 1)
      )
      
      // Sample question
      question = "What are the different chunking strategies mentioned in the document?"
      _ <- ZIO.logInfo(s"\nQuestion: $question")
      
      // Compare each chunking strategy
      _ <- ZIO.foreachDiscard(chunkers) { case (name, chunker) =>
        for
          _ <- ZIO.logInfo(s"\n--- $name Chunking ---")
          
          // Chunk the document
          chunks <- chunker.chunk(document)
          _ <- ZIO.logInfo(s"Created ${chunks.size} chunks")
          
          // Create embeddings for the chunks
          embeddedChunks <- embeddingModel.embedDocuments(chunks)
          _ <- ZIO.logInfo("Created embeddings for all chunks")
          
          // Create an in-memory retriever
          retriever = new InMemoryRetriever(embeddedChunks, embeddingModel)
          
          // Create a RAG chain
          ragChain = createRAGChain(llm, retriever)
          
          // Answer the question
          _ <- ZIO.logInfo("Generating answer...")
          answer <- ragChain.run(question)
          
          // Display the answer
          _ <- ZIO.logInfo(s"\nAnswer ($name chunking):")
          _ <- ZIO.logInfo(answer)
        yield ()
      }
    yield ()
  
  /**
   * Creates a RAG chain that uses a retriever and an LLM to answer questions.
   *
   * @param llm The LLM service
   * @param retriever The retriever service
   * @return A Chain that takes a question and produces an answer
   */
  private def createRAGChain(llm: LLM, retriever: Retriever): Chain[Any, LangChainError, String, String] =
    // Create a chain that retrieves relevant documents
    val retrievalChain = Chain[Any, LangChainError, String, Seq[Document]] { query =>
      retriever.retrieve(query, maxResults = 3)
        .mapError(e => e: LangChainError)
    }
    
    // Create a chain that formats the prompt with the retrieved documents
    val promptChain = Chain[Any, LangChainError, Seq[Document], String] { documents =>
      val docsText = documents.map(doc => s"Content: ${doc.content}").mkString("\n\n")
      ZIO.succeed(
        s"""Answer the question based only on the following context:
           |
           |$docsText
           |
           |Question: {{question}}
           |
           |Answer:""".stripMargin
      )
    }
    
    // Create a chain that replaces the {{question}} placeholder with the actual question
    val templateChain = Chain[Any, LangChainError, (String, String), String] { case (template, question) =>
      ZIO.succeed(template.replace("{{question}}", question))
    }
    
    // Create a chain that sends the prompt to the LLM
    val llmChain = Chain[Any, LangChainError, String, String] { prompt =>
      llm.complete(prompt)
        .mapError(e => e: LangChainError)
    }
    
    // Combine the chains
    Chain[Any, LangChainError, String, (String, Seq[Document])] { query =>
      retrievalChain.run(query).map(docs => (query, docs))
    } >>> Chain[Any, LangChainError, (String, Seq[Document]), (String, String)] { case (query, docs) =>
      promptChain.run(docs).map(template => (template, query))
    } >>> templateChain >>> llmChain
  
  /**
   * A simple in-memory retriever that finds relevant documents based on embedding similarity.
   *
   * @param documents The documents with their embeddings
   * @param embeddingModel The embedding model to use for query embedding
   */
  private class InMemoryRetriever(
    documents: Seq[(Document, Embedding)],
    embeddingModel: EmbeddingModel
  ) extends Retriever:
    override def retrieve(query: String, maxResults: Int): ZIO[Any, RetrieverError, Seq[Document]] =
      for
        // Embed the query
        queryEmbedding <- embeddingModel.embedQuery(query)
          .mapError(e => RetrieverError(e))
        
        // Calculate similarity scores
        similarities = documents.map { case (doc, embedding) =>
          (doc, queryEmbedding.cosineSimilarity(embedding))
        }
        
        // Sort by similarity and take the top results
        topResults = similarities
          .sortBy(-_._2) // Sort by descending similarity
          .take(maxResults)
          .map(_._1) // Take just the documents
      yield topResults