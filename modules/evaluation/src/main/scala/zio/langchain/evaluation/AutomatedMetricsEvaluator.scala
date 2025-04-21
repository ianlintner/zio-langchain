package zio.langchain.evaluation

import zio.*
import zio.langchain.core.domain.*

import scala.collection.immutable.Map
import scala.math.{log, min, max, sqrt}

/**
 * Evaluator that uses automated metrics to assess text quality without requiring an LLM.
 * Implements common NLP evaluation metrics like BLEU, ROUGE, etc.
 */
trait AutomatedMetricsEvaluator:
  /**
   * Evaluates text using automated metrics.
   *
   * @param reference The reference text (ground truth)
   * @param candidate The candidate text to evaluate
   * @param metrics The metrics to calculate (default: all available metrics)
   * @return A ZIO effect that produces an EvaluationResult or fails with an EvaluationError
   */
  def evaluate(
    reference: String,
    candidate: String,
    metrics: Set[AutomatedMetric] = AutomatedMetric.All
  ): ZIO[Any, EvaluationError, EvaluationResult]

/**
 * Companion object for AutomatedMetricsEvaluator.
 */
object AutomatedMetricsEvaluator:
  /**
   * Creates a new AutomatedMetricsEvaluator.
   *
   * @return A new AutomatedMetricsEvaluator
   */
  def make(): AutomatedMetricsEvaluator = new DefaultAutomatedMetricsEvaluator()

/**
 * Enumeration of available automated metrics.
 */
enum AutomatedMetric:
  case ExactMatch, F1Score, BLEU, ROUGE, Cosine, Jaccard, Levenshtein

/**
 * Companion object for AutomatedMetric.
 */
object AutomatedMetric:
  /**
   * Set of all available metrics.
   */
  val All: Set[AutomatedMetric] = Set(
    ExactMatch, F1Score, BLEU, ROUGE, Cosine, Jaccard, Levenshtein
  )

/**
 * Default implementation of AutomatedMetricsEvaluator.
 */
private class DefaultAutomatedMetricsEvaluator extends AutomatedMetricsEvaluator:
  override def evaluate(
    reference: String,
    candidate: String,
    metrics: Set[AutomatedMetric] = AutomatedMetric.All
  ): ZIO[Any, EvaluationError, EvaluationResult] =
    ZIO.attempt {
      // Calculate requested metrics
      val metricValues = metrics.flatMap { metric =>
        calculateMetric(reference, candidate, metric).map(value => (metric.toString.toLowerCase, value))
      }.toMap
      
      // Calculate overall score (average of all metrics)
      val overallScore = if metricValues.isEmpty then 0.0
                         else metricValues.values.sum / metricValues.size
      
      EvaluationResult(
        score = overallScore,
        metrics = metricValues
      )
    }.mapError(e => EvaluationError(e, "Failed to calculate automated metrics"))
  
  /**
   * Calculates a specific metric for the given reference and candidate texts.
   *
   * @param reference The reference text
   * @param candidate The candidate text
   * @param metric The metric to calculate
   * @return The calculated metric value
   */
  private def calculateMetric(
    reference: String,
    candidate: String,
    metric: AutomatedMetric
  ): Option[Double] =
    // Normalize and tokenize texts
    val refTokens = tokenize(reference)
    val candTokens = tokenize(candidate)
    
    // Calculate the requested metric
    metric match
      case AutomatedMetric.ExactMatch =>
        Some(if reference.trim == candidate.trim then 1.0 else 0.0)
        
      case AutomatedMetric.F1Score =>
        Some(calculateF1Score(refTokens, candTokens))
        
      case AutomatedMetric.BLEU =>
        Some(calculateBLEU(refTokens, candTokens))
        
      case AutomatedMetric.ROUGE =>
        Some(calculateROUGE(refTokens, candTokens))
        
      case AutomatedMetric.Cosine =>
        Some(calculateCosineSimilarity(refTokens, candTokens))
        
      case AutomatedMetric.Jaccard =>
        Some(calculateJaccardSimilarity(refTokens, candTokens))
        
      case AutomatedMetric.Levenshtein =>
        Some(calculateNormalizedLevenshtein(reference, candidate))
  
  /**
   * Tokenizes text into words.
   *
   * @param text The text to tokenize
   * @return A sequence of tokens
   */
  private def tokenize(text: String): Seq[String] =
    text.toLowerCase
      .replaceAll("[^\\p{L}\\p{N}\\s]", " ")
      .split("\\s+")
      .filter(_.nonEmpty)
      .toSeq
  
  /**
   * Calculates F1 score between reference and candidate texts.
   *
   * @param refTokens The reference tokens
   * @param candTokens The candidate tokens
   * @return The F1 score
   */
  private def calculateF1Score(refTokens: Seq[String], candTokens: Seq[String]): Double =
    val refSet = refTokens.toSet
    val candSet = candTokens.toSet
    
    val truePositives = refSet.intersect(candSet).size
    val precision = if candSet.isEmpty then 0.0 else truePositives.toDouble / candSet.size
    val recall = if refSet.isEmpty then 0.0 else truePositives.toDouble / refSet.size
    
    if precision + recall == 0.0 then 0.0
    else 2 * precision * recall / (precision + recall)
  
  /**
   * Calculates a simplified BLEU score.
   * This is a simplified version that focuses on n-gram precision with a brevity penalty.
   *
   * @param refTokens The reference tokens
   * @param candTokens The candidate tokens
   * @return The BLEU score
   */
  private def calculateBLEU(refTokens: Seq[String], candTokens: Seq[String]): Double =
    if candTokens.isEmpty then return 0.0
    if refTokens.isEmpty then return 0.0
    
    // Calculate n-gram precisions (for n=1 to 4)
    val maxN = min(4, min(refTokens.length, candTokens.length))
    val nGramPrecisions = (1 to maxN).map { n =>
      val candNGrams = getNGrams(candTokens, n)
      val refNGrams = getNGrams(refTokens, n)
      
      if candNGrams.isEmpty then 0.0
      else {
        val matches = candNGrams.count(ngram => refNGrams.contains(ngram))
        matches.toDouble / candNGrams.size
      }
    }
    
    // If all precisions are 0, return 0
    if nGramPrecisions.forall(_ == 0.0) then return 0.0
    
    // Calculate geometric mean of precisions
    val geometricMean = nGramPrecisions.map(p => if p > 0 then log(p) else 0.0).sum / nGramPrecisions.size
    val score = math.exp(geometricMean)
    
    // Apply brevity penalty
    val brevityPenalty = if candTokens.length >= refTokens.length then 1.0
                         else math.exp(1 - refTokens.length.toDouble / candTokens.length)
    
    brevityPenalty * score
  
  /**
   * Gets n-grams from a sequence of tokens.
   *
   * @param tokens The tokens
   * @param n The n-gram size
   * @return A sequence of n-grams
   */
  private def getNGrams(tokens: Seq[String], n: Int): Seq[Seq[String]] =
    if tokens.length < n then Seq.empty
    else tokens.sliding(n).toSeq
  
  /**
   * Calculates a simplified ROUGE score.
   * This is a simplified version that focuses on recall of n-grams.
   *
   * @param refTokens The reference tokens
   * @param candTokens The candidate tokens
   * @return The ROUGE score
   */
  private def calculateROUGE(refTokens: Seq[String], candTokens: Seq[String]): Double =
    if refTokens.isEmpty then return 0.0
    if candTokens.isEmpty then return 0.0
    
    // Calculate ROUGE-1 (unigram recall)
    val refUnigrams = refTokens.groupBy(identity).view.mapValues(_.size).toMap
    val candUnigrams = candTokens.groupBy(identity).view.mapValues(_.size).toMap
    
    val matchingUnigrams = refUnigrams.map { case (token, refCount) =>
      val candCount = candUnigrams.getOrElse(token, 0)
      min(refCount, candCount)
    }.sum
    
    val recall = matchingUnigrams.toDouble / refTokens.size
    
    // Calculate ROUGE-2 (bigram recall) if possible
    val rouge2 = if refTokens.length >= 2 && candTokens.length >= 2 then
      val refBigrams = getNGrams(refTokens, 2).groupBy(identity).view.mapValues(_.size).toMap
      val candBigrams = getNGrams(candTokens, 2).groupBy(identity).view.mapValues(_.size).toMap
      
      val matchingBigrams = refBigrams.map { case (bigram, refCount) =>
        val candCount = candBigrams.getOrElse(bigram, 0)
        min(refCount, candCount)
      }.sum
      
      val bigramRecall = matchingBigrams.toDouble / getNGrams(refTokens, 2).size
      bigramRecall
    else 0.0
    
    // Average of ROUGE-1 and ROUGE-2
    (recall + rouge2) / 2
  
  /**
   * Calculates cosine similarity between reference and candidate texts.
   *
   * @param refTokens The reference tokens
   * @param candTokens The candidate tokens
   * @return The cosine similarity
   */
  private def calculateCosineSimilarity(refTokens: Seq[String], candTokens: Seq[String]): Double =
    if refTokens.isEmpty || candTokens.isEmpty then return 0.0
    
    // Create term frequency vectors
    val allTerms = (refTokens ++ candTokens).distinct
    val refVector = allTerms.map(term => refTokens.count(_ == term).toDouble)
    val candVector = allTerms.map(term => candTokens.count(_ == term).toDouble)
    
    // Calculate cosine similarity
    val dotProduct = (refVector zip candVector).map { case (a, b) => a * b }.sum
    val refMagnitude = sqrt(refVector.map(x => x * x).sum)
    val candMagnitude = sqrt(candVector.map(x => x * x).sum)
    
    if refMagnitude == 0 || candMagnitude == 0 then 0.0
    else dotProduct / (refMagnitude * candMagnitude)
  
  /**
   * Calculates Jaccard similarity between reference and candidate texts.
   *
   * @param refTokens The reference tokens
   * @param candTokens The candidate tokens
   * @return The Jaccard similarity
   */
  private def calculateJaccardSimilarity(refTokens: Seq[String], candTokens: Seq[String]): Double =
    val refSet = refTokens.toSet
    val candSet = candTokens.toSet
    
    if refSet.isEmpty && candSet.isEmpty then 1.0
    else if refSet.isEmpty || candSet.isEmpty then 0.0
    else refSet.intersect(candSet).size.toDouble / refSet.union(candSet).size
  
  /**
   * Calculates normalized Levenshtein distance between reference and candidate texts.
   * The result is inverted (1 - distance) so that higher values indicate better similarity.
   *
   * @param reference The reference text
   * @param candidate The candidate text
   * @return The normalized Levenshtein similarity
   */
  private def calculateNormalizedLevenshtein(reference: String, candidate: String): Double =
    val maxLen = max(reference.length, candidate.length)
    if maxLen == 0 then return 1.0
    
    val distance = levenshteinDistance(reference, candidate)
    1.0 - distance.toDouble / maxLen
  
  /**
   * Calculates Levenshtein distance between two strings.
   *
   * @param s1 The first string
   * @param s2 The second string
   * @return The Levenshtein distance
   */
  private def levenshteinDistance(s1: String, s2: String): Int =
    val m = s1.length
    val n = s2.length
    
    // Create a matrix of size (m+1) x (n+1)
    val d = Array.ofDim[Int](m + 1, n + 1)
    
    // Initialize the first row and column
    for i <- 0 to m do d(i)(0) = i
    for j <- 0 to n do d(0)(j) = j
    
    // Fill the matrix
    for j <- 1 to n do
      for i <- 1 to m do
        val cost = if s1(i - 1) == s2(j - 1) then 0 else 1
        d(i)(j) = min(min(d(i - 1)(j) + 1, d(i)(j - 1) + 1), d(i - 1)(j - 1) + cost)
    
    d(m)(n)