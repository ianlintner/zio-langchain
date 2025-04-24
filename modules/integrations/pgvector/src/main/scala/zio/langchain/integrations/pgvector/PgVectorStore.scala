package zio.langchain.integrations.pgvector

import zio.*
import zio.json.*
import zio.stream.ZStream

import zio.langchain.core.retriever.Retriever
import zio.langchain.core.model.EmbeddingModel
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

import java.sql.{Connection, DriverManager, PreparedStatement, ResultSet, SQLException}
import java.util.UUID
import java.util.Properties
import javax.sql.DataSource
import scala.collection.mutable.ArrayBuffer

/**
 * Implementation of a vector store using PostgreSQL with pgvector extension.
 * This class provides methods for storing and retrieving documents using vector embeddings.
 *
 * @param config The PgVector configuration
 * @param embeddingModel The embedding model to use for generating embeddings
 * @param dataSource The data source for database connections
 */
class PgVectorStore private (
  config: PgVectorConfig,
  embeddingModel: EmbeddingModel,
  private val dataSource: DataSource
) extends Retriever:
  import PgVectorStore.*
  import zio.langchain.core.errors.PgVectorError

  /**
   * Initializes the database schema and tables if they don't exist.
   * This method creates the necessary extension, schema, and table for storing document embeddings.
   *
   * @return A ZIO effect that completes when the initialization is done
   */
  def initialize: ZIO[Any, RetrieverError, Unit] =
    withConnection { conn =>
      ZIO.attempt {
        val stmt = conn.createStatement()
        
        // Create pgvector extension if it doesn't exist
        val _ = stmt.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        // Create schema if it doesn't exist
        if (config.schema != "public") {
          val _ = stmt.execute(s"CREATE SCHEMA IF NOT EXISTS ${config.schema}")
        }
        
        // Create table if it doesn't exist
        val createTableSql =
          s"""
          CREATE TABLE IF NOT EXISTS ${config.schema}.${config.table} (
            ${config.idColumn} TEXT PRIMARY KEY,
            ${config.contentColumn} TEXT NOT NULL,
            ${config.vectorColumn} vector(${config.dimension}) NOT NULL,
            ${config.metadataColumn} JSONB
          )
          """
        val _ = stmt.execute(createTableSql)
        
        // Create index for vector similarity search if it doesn't exist
        val indexName = s"${config.table}_${config.vectorColumn}_idx"
        val indexMethod = config.distanceType match {
          case "cosine" => "cosine_ops"
          case "l2" => "l2_ops"
          case "inner" => "inner_product_ops"
          case _ => "cosine_ops" // Default to cosine
        }
        
        val createIndexSql =
          s"""
          CREATE INDEX IF NOT EXISTS ${indexName} ON ${config.schema}.${config.table}
          USING ivfflat (${config.vectorColumn} ${indexMethod})
          """
        val _ = stmt.execute(createIndexSql)
      }.mapError {
        case e: SQLException =>
          PgVectorError.connectionError(s"Failed to initialize database: ${e.getMessage}")
        case e =>
          PgVectorError.unknownError(e)
      }
    }

  /**
   * Adds documents to the PostgreSQL table.
   *
   * @param documents The documents to add
   * @return A ZIO effect that completes when the documents are added
   */
  def addDocuments(documents: Seq[Document]): ZIO[Any, RetrieverError, Unit] =
    if (documents.isEmpty) ZIO.unit
    else
      for {
        // Generate embeddings for the documents
        docEmbeddings <- embeddingModel.embedDocuments(documents)
          .mapError(err => RetrieverError(err, "Failed to generate embeddings for documents"))
        
        // Validate dimensions match the configured dimension
        _ <- ZIO.foreach(docEmbeddings) { case (_, embedding) =>
          if (embedding.values.length != config.dimension)
            ZIO.fail(PgVectorError.dimensionMismatchError(config.dimension, embedding.values.length))
          else
            ZIO.unit
        }
        
        // Insert documents into the database
        _ <- withConnection { conn =>
          ZIO.attempt {
            val insertSql = 
              s"""
              INSERT INTO ${config.schema}.${config.table} 
              (${config.idColumn}, ${config.contentColumn}, ${config.vectorColumn}, ${config.metadataColumn})
              VALUES (?, ?, ?::vector, ?::jsonb)
              ON CONFLICT (${config.idColumn}) DO UPDATE SET
              ${config.contentColumn} = EXCLUDED.${config.contentColumn},
              ${config.vectorColumn} = EXCLUDED.${config.vectorColumn},
              ${config.metadataColumn} = EXCLUDED.${config.metadataColumn}
              """
            
            val prepStmt = conn.prepareStatement(insertSql)
            
            // Use batch insert for better performance
            docEmbeddings.foreach { case (doc, embedding) =>
              prepStmt.setString(1, doc.id)
              prepStmt.setString(2, doc.content)
              prepStmt.setString(3, embedding.values.mkString("[", ",", "]"))
              prepStmt.setString(4, doc.metadata.toJson)
              prepStmt.addBatch()
            }
            
            prepStmt.executeBatch()
            prepStmt.close()
          }.mapError {
            case e: SQLException => 
              PgVectorError.queryError(s"Failed to add documents: ${e.getMessage}")
            case e => 
              PgVectorError.unknownError(e)
          }
        }
      } yield ()

  /**
   * Adds a single document to the PostgreSQL table.
   *
   * @param document The document to add
   * @return A ZIO effect that completes when the document is added
   */
  def addDocument(document: Document): ZIO[Any, RetrieverError, Unit] =
    addDocuments(Seq(document))

  /**
   * Deletes documents from the PostgreSQL table by their IDs.
   *
   * @param ids The IDs of the documents to delete
   * @return A ZIO effect that completes when the documents are deleted
   */
  def deleteDocuments(ids: Seq[String]): ZIO[Any, RetrieverError, Unit] =
    if (ids.isEmpty) ZIO.unit
    else
      withConnection { conn =>
        ZIO.attempt {
          val placeholders = ids.map(_ => "?").mkString(",")
          val deleteSql = 
            s"""
            DELETE FROM ${config.schema}.${config.table}
            WHERE ${config.idColumn} IN ($placeholders)
            """
          
          val prepStmt = conn.prepareStatement(deleteSql)
          
          ids.zipWithIndex.foreach { case (id, idx) =>
            prepStmt.setString(idx + 1, id)
          }
          
          prepStmt.executeUpdate()
          prepStmt.close()
        }.mapError {
          case e: SQLException => 
            PgVectorError.queryError(s"Failed to delete documents: ${e.getMessage}")
          case e => 
            PgVectorError.unknownError(e)
        }
      }

  /**
   * Deletes all documents from the PostgreSQL table.
   *
   * @return A ZIO effect that completes when all documents are deleted
   */
  def deleteAll(): ZIO[Any, RetrieverError, Unit] =
    withConnection { conn =>
      ZIO.attempt {
        val deleteSql = s"DELETE FROM ${config.schema}.${config.table}"
        val stmt = conn.createStatement()
        stmt.executeUpdate(deleteSql)
        stmt.close()
      }.mapError {
        case e: SQLException => 
          PgVectorError.queryError(s"Failed to delete all documents: ${e.getMessage}")
        case e => 
          PgVectorError.unknownError(e)
      }
    }

  /**
   * Retrieves documents relevant to a query.
   *
   * @param query The query string
   * @param maxResults The maximum number of results to return
   * @return A ZIO effect that produces a sequence of documents
   */
  override def retrieve(query: String, maxResults: Int = 10): ZIO[Any, RetrieverError, Seq[Document]] =
    retrieveWithScores(query, maxResults).map(_.map(_._1))

  /**
   * Retrieves documents relevant to a query with their similarity scores.
   *
   * @param query The query string
   * @param maxResults The maximum number of results to return
   * @return A ZIO effect that produces a sequence of document-score pairs
   */
  override def retrieveWithScores(query: String, maxResults: Int = 10): ZIO[Any, RetrieverError, Seq[(Document, Double)]] =
    for {
      // Generate embedding for the query
      queryEmbedding <- embeddingModel.embedQuery(query)
        .mapError(err => RetrieverError(err, "Failed to generate embedding for query"))
      
      // Validate dimension
      _ <- ZIO.when(queryEmbedding.values.length != config.dimension) {
        ZIO.fail(PgVectorError.dimensionMismatchError(config.dimension, queryEmbedding.values.length))
      }
      
      // Perform similarity search
      results <- withConnection { conn =>
        ZIO.attempt {
          // Construct the similarity search query based on the distance type
          val distanceFunction = config.distanceType match {
            case "cosine" => s"1 - (${config.vectorColumn} <=> ?::vector)"
            case "l2" => s"1 / (1 + (${config.vectorColumn} <-> ?::vector))"
            case "inner" => s"${config.vectorColumn} <#> ?::vector"
            case _ => s"1 - (${config.vectorColumn} <=> ?::vector)" // Default to cosine
          }
          
          val searchSql = 
            s"""
            SELECT ${config.idColumn}, ${config.contentColumn}, ${config.metadataColumn}, 
                   $distanceFunction AS similarity
            FROM ${config.schema}.${config.table}
            ORDER BY similarity DESC
            LIMIT ?
            """
          
          val prepStmt = conn.prepareStatement(searchSql)
          prepStmt.setString(1, queryEmbedding.values.mkString("[", ",", "]"))
          prepStmt.setInt(2, maxResults)
          
          val rs = prepStmt.executeQuery()
          val results = new ArrayBuffer[(Document, Double)]()
          
          while (rs.next()) {
            val id = rs.getString(1)
            val content = rs.getString(2)
            val metadataJson = rs.getString(3)
            val similarity = rs.getDouble(4)
            
            val metadata = if (metadataJson != null) {
              metadataJson.fromJson[Map[String, String]].getOrElse(Map.empty)
            } else {
              Map.empty[String, String]
            }
            
            val doc = Document(id, content, metadata)
            results.append((doc, similarity))
          }
          
          rs.close()
          prepStmt.close()
          
          results.toSeq
        }.mapError {
          case e: SQLException => 
            PgVectorError.queryError(s"Failed to retrieve documents: ${e.getMessage}")
          case e => 
            PgVectorError.unknownError(e)
        }
      }
    } yield results

  /**
   * Helper method to execute a function with a database connection.
   *
   * @param f The function to execute with the connection
   * @return A ZIO effect that produces the result of the function
   */
  private def withConnection[R](f: Connection => ZIO[Any, RetrieverError, R]): ZIO[Any, RetrieverError, R] =
    ZIO.acquireReleaseWith(
      ZIO.attempt(dataSource.getConnection()).mapError(e => PgVectorError.connectionError(e.getMessage))
    )(conn =>
      ZIO.attempt {
        if (!conn.isClosed) conn.close()
      }.catchAll(_ => ZIO.unit)
    )(f)
    
  /**
   * Closes the underlying data source if it's AutoCloseable.
   * This method is used for resource cleanup.
   */
  def closeDataSource: ZIO[Any, Nothing, Unit] =
    ZIO.attempt {
      if (dataSource.isInstanceOf[AutoCloseable]) {
        dataSource.asInstanceOf[AutoCloseable].close()
      }
    }.catchAll(_ => ZIO.unit)

/**
 * Companion object for PgVectorStore.
 */
object PgVectorStore:
  /**
   * Creates a PgVectorStore with a connection pool.
   *
   * @param config The PgVector configuration
   * @param embeddingModel The embedding model to use
   * @return A ZIO effect that produces a PgVectorStore
   */
  def make(
    config: PgVectorConfig,
    embeddingModel: EmbeddingModel
  ): ZIO[Any, RetrieverError, PgVectorStore] =
    for {
      // Create a HikariCP data source
      dataSource <- ZIO.attempt {
        import com.zaxxer.hikari.{HikariConfig, HikariDataSource}
        
        val hikariConfig = new HikariConfig()
        hikariConfig.setJdbcUrl(config.jdbcUrl)
        hikariConfig.setUsername(config.username)
        hikariConfig.setPassword(config.password)
        hikariConfig.setMaximumPoolSize(config.connectionPoolSize)
        hikariConfig.setConnectionTimeout(config.timeout.toMillis)
        
        new HikariDataSource(hikariConfig)
      }.mapError(e => PgVectorError.connectionError(s"Failed to create connection pool: ${e.getMessage}"))
      
      // Create the PgVectorStore
      store = new PgVectorStore(config, embeddingModel, dataSource)
      
      // Initialize the database schema and tables
      _ <- store.initialize
    } yield store

  /**
   * Creates a ZLayer that provides a Retriever implementation using PgVector.
   *
   * @return A ZLayer that requires a PgVectorConfig and an EmbeddingModel and provides a Retriever
   */
  val live: ZLayer[PgVectorConfig & EmbeddingModel, RetrieverError, Retriever] =
    ZLayer.fromZIO(
      for {
        config <- ZIO.service[PgVectorConfig]
        embeddingModel <- ZIO.service[EmbeddingModel]
        store <- make(config, embeddingModel)
      } yield store
    )

  /**
   * Creates a ZLayer that provides a PgVectorStore.
   *
   * @return A ZLayer that requires a PgVectorConfig and an EmbeddingModel and provides a PgVectorStore
   */
  val liveStore: ZLayer[PgVectorConfig & EmbeddingModel, RetrieverError, PgVectorStore] =
    ZLayer.fromZIO(
      for {
        config <- ZIO.service[PgVectorConfig]
        embeddingModel <- ZIO.service[EmbeddingModel]
        store <- make(config, embeddingModel)
      } yield store
    )

  /**
   * Creates a scoped ZLayer that provides a PgVectorStore.
   * This ensures proper resource cleanup when the scope ends.
   *
   * @return A ZLayer that requires a PgVectorConfig and an EmbeddingModel and provides a PgVectorStore
   */
  val scoped: ZLayer[PgVectorConfig & EmbeddingModel, RetrieverError, PgVectorStore] =
    ZLayer.scoped {
      for {
        config <- ZIO.service[PgVectorConfig]
        embeddingModel <- ZIO.service[EmbeddingModel]
        store <- ZIO.acquireRelease(
          make(config, embeddingModel)
        )(store => store.closeDataSource)
      } yield store
    }