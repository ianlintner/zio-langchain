package zio.langchain.core.chain

import zio.*

import zio.langchain.core.errors.*

/**
 * Interface for chains.
 * Chains are composable operations that can be combined to form complex workflows.
 *
 * @tparam R The environment type required by this chain
 * @tparam E The error type that can be produced by this chain
 * @tparam I The input type accepted by this chain
 * @tparam O The output type produced by this chain
 */
trait Chain[-R, +E <: LangChainError, -I, +O]:
  /**
   * Runs the chain with the given input.
   *
   * @param input The input to the chain
   * @return A ZIO effect that requires an environment R, produces an output O, or fails with an error E
   */
  def run(input: I): ZIO[R, E, O]
  
  /**
   * Composes this chain with another chain, where the output of this chain is passed as input to the next chain.
   *
   * @param next The next chain to compose with
   * @tparam R1 The environment type required by the next chain
   * @tparam E1 The error type that can be produced by the next chain
   * @tparam O2 The output type produced by the next chain
   * @return A new chain that represents the composition of this chain with the next chain
   */
  def andThen[R1 <: R, E1 >: E <: LangChainError, O2](next: Chain[R1, E1, O, O2]): Chain[R1, E1, I, O2] =
    Chain.sequence(this, next)
  
  /**
   * Alias for andThen.
   */
  def >>>[R1 <: R, E1 >: E <: LangChainError, O2](next: Chain[R1, E1, O, O2]): Chain[R1, E1, I, O2] =
    andThen(next)
  
  /**
   * Maps the output of this chain using the given function.
   *
   * @param f The function to apply to the output
   * @tparam O2 The new output type
   * @return A new chain that applies the function to the output of this chain
   */
  def map[O2](f: O => O2): Chain[R, E, I, O2] =
    Chain.Map(this, f)
  
  /**
   * Recovers from errors produced by this chain using the given function.
   *
   * @param f The function to apply to the error
   * @tparam O2 The new output type
   * @return A new chain that applies the function to the error of this chain
   */
  def recover[O2 >: O](f: E => O2): Chain[R, Nothing, I, O2] =
    Chain.Recover(this, f)

/**
 * Companion object for Chain.
 */
object Chain:
  /**
   * Creates a new chain from a function.
   *
   * @param f The function to create a chain from
   * @tparam R The environment type required by the function
   * @tparam E The error type that can be produced by the function
   * @tparam I The input type accepted by the function
   * @tparam O The output type produced by the function
   * @return A new chain that wraps the function
   */
  def apply[R, E <: LangChainError, I, O](f: I => ZIO[R, E, O]): Chain[R, E, I, O] =
    new Chain[R, E, I, O]:
      override def run(input: I): ZIO[R, E, O] = f(input)
  
  /**
   * Creates a new chain that always produces the same output.
   *
   * @param value The output value
   * @tparam O The output type
   * @return A new chain that always produces the given value
   */
  def succeed[O](value: O): Chain[Any, Nothing, Any, O] =
    new Chain[Any, Nothing, Any, O]:
      override def run(input: Any): ZIO[Any, Nothing, O] = ZIO.succeed(value)
  
  /**
   * Creates a new chain that always fails with the same error.
   *
   * @param error The error value
   * @tparam E The error type
   * @return A new chain that always fails with the given error
   */
  def fail[E <: LangChainError](error: E): Chain[Any, E, Any, Nothing] =
    new Chain[Any, E, Any, Nothing]:
      override def run(input: Any): ZIO[Any, E, Nothing] = ZIO.fail(error)
  
  /**
   * Creates a new chain that passes the input through unchanged.
   *
   * @tparam I The input/output type
   * @return A new chain that passes the input through unchanged
   */
  def identity[I]: Chain[Any, Nothing, I, I] =
    new Chain[Any, Nothing, I, I]:
      override def run(input: I): ZIO[Any, Nothing, I] = ZIO.succeed(input)
  
  /**
   * Composes two chains, where the output of the first chain is passed as input to the second chain.
   *
   * @param first The first chain
   * @param second The second chain
   * @tparam R The combined environment type required by both chains
   * @tparam E The combined error type that can be produced by both chains
   * @tparam I The input type accepted by the first chain
   * @tparam O The intermediate output type produced by the first chain and accepted by the second chain
   * @tparam O2 The final output type produced by the second chain
   * @return A new chain that represents the composition of the two chains
   */
  def sequence[R, E <: LangChainError, I, O, O2](
    first: Chain[R, E, I, O],
    second: Chain[R, E, O, O2]
  ): Chain[R, E, I, O2] = new Chain[R, E, I, O2]:
    override def run(input: I): ZIO[R, E, O2] =
      first.run(input).flatMap(second.run)
  
  /**
   * A chain that maps the output of another chain using a function.
   *
   * @param chain The chain to map
   * @param f The function to apply to the output
   * @tparam R The environment type required by the chain
   * @tparam E The error type that can be produced by the chain
   * @tparam I The input type accepted by the chain
   * @tparam O The output type produced by the chain
   * @tparam O2 The new output type after applying the function
   */
  private case class Map[R, E <: LangChainError, I, O, O2](
    chain: Chain[R, E, I, O],
    f: O => O2
  ) extends Chain[R, E, I, O2]:
    override def run(input: I): ZIO[R, E, O2] =
      chain.run(input).map(f)
  
  /**
   * A chain that recovers from errors produced by another chain using a function.
   *
   * @param chain The chain to recover
   * @param f The function to apply to the error
   * @tparam R The environment type required by the chain
   * @tparam E The error type that can be produced by the chain
   * @tparam I The input type accepted by the chain
   * @tparam O The output type produced by the chain
   */
  private case class Recover[R, E <: LangChainError, I, O](
    chain: Chain[R, E, I, O],
    f: E => O
  ) extends Chain[R, Nothing, I, O]:
    override def run(input: I): ZIO[R, Nothing, O] =
      chain.run(input).catchAll(e => ZIO.succeed(f(e)))