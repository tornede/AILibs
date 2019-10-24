package ai.libs.jaicore.ml.core.olddataset.attribute.transformer.multivalue;

import static org.junit.Assert.assertArrayEquals;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.junit.Test;

import ai.libs.jaicore.ml.core.dataset.schema.attribute.MultiValueAttribute;
import ai.libs.jaicore.ml.core.dataset.schema.attribute.MultiValueAttributeValue;
import ai.libs.jaicore.ml.core.olddataset.attributetransformer.multivalue.MinHashingTransformer;

public class MinHashingTransformerTest {

	private static final double DELTA = 0.00001;

	@Test
	public void testCreatedSignatures() {
		Set<String> domain = new HashSet<>();
		domain.add("a");
		domain.add("b");
		domain.add("c");
		domain.add("d");
		domain.add("e");
		MultiValueAttribute type = new MultiValueAttribute("a0", domain);

		MultiValueAttributeValue v1 = new MultiValueAttributeValue(type, Arrays.asList("a", "b", "c", "d", "e"));
		MultiValueAttributeValue v2 = new MultiValueAttributeValue(type, Arrays.asList("a", "c", "d", "e"));

		MinHashingTransformer transformer = new MinHashingTransformer(5, 3, 1l);
		double[] h1 = transformer.transformAttribute(v1);
		double[] h2 = transformer.transformAttribute(v2);

		assertArrayEquals(h1, h2, DELTA);

	}

}