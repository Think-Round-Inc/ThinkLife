
import asyncio
import time

from brain.data_sources.vector_db import create_vector_data_source


async def main():
    source = create_vector_data_source()

    print("Initializing Zoe VectorDataSource...")

    initialized = await source.initialize()
    if not initialized:
        raise RuntimeError(
            "Initialization failed. Check ChromaDB replicas, "
            "embedding model, and data source registration."
        )

    try:
        questions = [
            "What is ThinkLife?",
            "How does Zoe work?",
            "What is empathy?",
            "What is ThinkLife?",
            "How does Zoe work?",
        ]

        for number, question in enumerate(questions, start=1):
            print(f"\n{'=' * 55}")
            print(f"Query {number}: {question}")

            before = source.cache_stats()
            start = time.perf_counter()

            results = await source.query(
                question,
                k=5,
                mode="hybrid"
            )

            latency = (time.perf_counter() - start) * 1000
            after = source.cache_stats()

            if after["hits"] > before["hits"]:
                status = "CACHE HIT"
            elif after["misses"] > before["misses"]:
                status = "CACHE MISS"
            else:
                status = "CACHE NOT USED"

            print(f"Status: {status}")
            print(f"Latency: {latency:.2f} ms")
            print(f"Documents returned: {len(results)}")
            print(f"Cache hits: {after['hits']}")
            print(f"Cache misses: {after['misses']}")
            print(f"Cache entries: {after['entries']}")
            print(f"Cache hit rate: {after['hit_rate']:.1%}")

        print("\nFINAL CACHE STATISTICS")
        print(source.cache_stats())

    finally:
        await source.close()


if __name__ == "__main__":
    asyncio.run(main())
