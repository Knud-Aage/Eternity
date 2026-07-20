package dk.puzzle.model;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the {@link CandidateValidator} contract.
 *
 * CandidateValidator is a pure interface with no implementation of its own —
 * there is no logic in the source file to branch-cover. What can and should
 * be verified is the shape of the contract it defines: it must remain a
 * single-abstract-method (lambda-compatible) interface with the documented
 * signature, and any implementation plugged in must receive exactly the
 * arguments the caller passed and be free to return an arbitrary filtered
 * list. These tests exercise that contract via small inline implementations.
 */
class CandidateValidatorTest {

    @Test
    void testInterfaceHasExactlyOneAbstractMethodNamedValidate() {
        Method[] abstractMethods = Arrays.stream(CandidateValidator.class.getMethods())
                .filter(m -> Modifier.isAbstract(m.getModifiers()))
                .toArray(Method[]::new);

        assertEquals(1, abstractMethods.length,
                "CandidateValidator must stay a functional (single-abstract-method) interface so lambdas keep compiling");
        assertEquals("validate", abstractMethods[0].getName());
    }

    @Test
    void testValidateMethodSignatureMatchesDocumentedContract() throws NoSuchMethodException {
        Method validate = CandidateValidator.class.getDeclaredMethod("validate", int[].class, int.class);

        assertEquals(List.class, validate.getReturnType(), "validate must return a List<int[]>");
        assertEquals(2, validate.getParameterCount());
        assertEquals(int[].class, validate.getParameterTypes()[0], "First parameter is the flattened candidate batch");
        assertEquals(int.class, validate.getParameterTypes()[1], "Second parameter is the permutation count");
    }

    @Test
    void testLambdaImplementationReceivesExactBatchReferenceAndCount() {
        int[] batch = {1, 2, 3, 4};
        AtomicReference<int[]> receivedBatch = new AtomicReference<>();
        AtomicInteger receivedCount = new AtomicInteger(-1);

        CandidateValidator validator = (candidateBatch, numPermutations) -> {
            receivedBatch.set(candidateBatch);
            receivedCount.set(numPermutations);
            return Collections.emptyList();
        };

        List<int[]> result = validator.validate(batch, 7);

        assertSame(batch, receivedBatch.get(), "Implementations must receive the exact array reference, not a copy");
        assertEquals(7, receivedCount.get());
        assertTrue(result.isEmpty());
    }

    @Test
    void testImplementationCanPartitionBatchIntoSixteenPieceCandidates() {
        // Mirrors the documented usage: a flattened batch of 16-piece macro-tiles,
        // sliced into numPermutations arrays of length 16.
        CandidateValidator validator = (candidateBatch, numPermutations) -> {
            List<int[]> out = new ArrayList<>();
            for (int i = 0; i < numPermutations; i++) {
                out.add(Arrays.copyOfRange(candidateBatch, i * 16, i * 16 + 16));
            }
            return out;
        };

        int[] batch = new int[32];
        for (int i = 0; i < 32; i++) {
            batch[i] = i;
        }

        List<int[]> result = validator.validate(batch, 2);

        assertEquals(2, result.size());
        assertArrayEquals(Arrays.copyOfRange(batch, 0, 16), result.get(0));
        assertArrayEquals(Arrays.copyOfRange(batch, 16, 32), result.get(1));
    }

    @Test
    void testImplementationMayFilterOutInvalidCandidatesAndReturnFewerThanRequested() {
        // "Validated" candidates need not equal numPermutations in count; the
        // interface only guarantees a subset of internally-consistent tiles.
        CandidateValidator validator = (candidateBatch, numPermutations) -> {
            List<int[]> out = new ArrayList<>();
            for (int i = 0; i < numPermutations; i++) {
                int[] tile = Arrays.copyOfRange(candidateBatch, i * 4, i * 4 + 4);
                if (tile[0] != 0) { // arbitrary "valid" rule for this test double
                    out.add(tile);
                }
            }
            return out;
        };

        int[] batch = {0, 0, 0, 0, 9, 1, 2, 3};

        List<int[]> result = validator.validate(batch, 2);

        assertEquals(1, result.size(), "Implementations are free to drop candidates that fail their own validity check");
        assertArrayEquals(new int[]{9, 1, 2, 3}, result.get(0));
    }

    @Test
    void testValidateWithZeroPermutationsOnEmptyBatchReturnsEmptyList() {
        CandidateValidator validator = (candidateBatch, numPermutations) -> {
            List<int[]> out = new ArrayList<>();
            for (int i = 0; i < numPermutations; i++) {
                out.add(Arrays.copyOfRange(candidateBatch, i * 16, i * 16 + 16));
            }
            return out;
        };

        List<int[]> result = validator.validate(new int[0], 0);

        assertNotNull(result);
        assertTrue(result.isEmpty(), "Zero permutations over an empty batch must produce an empty (not null) result");
    }
}
