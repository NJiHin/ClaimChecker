from google import genai
import asyncpg

_PROMPT = """\
You are a claim verification assistant. Given context passages and a claim, decide if the claim is supported by the context. Respond with exactly one word: "Verified" or "Unverified".

Rules:
- Respond "Verified" if the context supports the claim, even if the wording differs. The claim does not need to be a verbatim quote — paraphrases, summaries, and reasonable inferences from the context all count as supported.
- Respond "Unverified" if the context contradicts the claim, or contains no information relevant to the claim.
- Do NOT use any outside knowledge. Base your decision strictly on the provided context.
- Output should be "Verified" or "Unverified" ONLY

---

Example 1:
Context:
- The United States imported $539 billion worth of goods from China in 2018.
- The U.S. trade deficit with China reached $419 billion in 2018.

Claim: The US trade deficit with China was $419 billion in 2018.

Answer: Verified

---

Example 2:
Context:
- China's GDP growth slowed to 6.6% in 2018.
- The Chinese government implemented stimulus measures to support the economy.

Claim: China's GDP growth was 8% in 2018.

Answer: Unverified

---

Example 3:
Context:
- Bilateral talks between the US and China resumed in January 2019.
- Both sides agreed to a 90-day negotiation period.

Claim: The US imposed sanctions on Chinese tech companies in 2019.

Answer: Unverified

---

Example 4:
Context:
- With import and export shares of 2017 GDP of 17.9% and 19.7%, respectively, the trade war affected transactions equivalent to about 5.5% of China's GDP.

Claim: The trade conflict targets transactions equivalent to roughly 5.5% of China's GDP.

Answer: Verified

---

Example 5:
Context:
- China accounted for an average 23% of US imports, and the US accounted for an average 12% of Chinese imports.

Claim: China accounts for 23% of all US imports.

Answer: Verified

---

Now verify the following:
Context:
{chunks}

Claim: {claim}

Answer:"""

_SIMILARITY_THRESHOLD = 0.55
_TOP_K = 5

async def _get_chunks(
        sentence: str,
        conn: asyncpg.Connection,
        client: genai.Client,
) -> str:
    result = await client.aio.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[sentence],
    )
    query_vector = result.embeddings[0].values
    vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

    rows = await conn.fetch(
        """
        SELECT text, 1 - (embedding <=> $1::vector) AS similarity
        FROM passages
        WHERE 1 - (embedding <=> $1::vector) >= $2
        ORDER BY similarity DESC
        LIMIT $3
        """,
        vector_str,
        _SIMILARITY_THRESHOLD,
        _TOP_K,
    )

    if not rows:
        return ""

    return "\n".join(f"- {row['text']}" for row in rows)

async def verify(
        sentence: str,
        pool: asyncpg.Pool,
        client: genai.Client,
) -> str:
    async with pool.acquire() as conn:
        chunks = await _get_chunks(sentence=sentence, conn=conn, client=client)

    if not chunks:
        return "Unverified"

    prompt = _PROMPT.format(chunks=chunks, claim=sentence)
    verdict = await client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt],
        config=genai.types.GenerateContentConfig(
            max_output_tokens=100,
            temperature=0.0
        )
    )

    return verdict.text.strip() if verdict.text else "Unverified"