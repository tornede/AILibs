package ai.libs.jaicore.ml.core.dataset.schema.attribute;

import java.util.Collection;

import org.api4.java.ai.ml.core.dataset.schema.attribute.IAttribute;
import org.api4.java.ai.ml.core.dataset.schema.attribute.IMultiLabelAttributeValue;

public class MultiValueAttributeValue implements IMultiLabelAttributeValue {

	private MultiValueAttribute attribute;
	private Collection<String> value;

	public MultiValueAttributeValue(final MultiValueAttribute attribute, final Collection<String> value) {
		this.attribute = attribute;
		this.value = value;
	}

	@Override
	public Collection<String> getValue() {
		return this.value;
	}

	@Override
	public IAttribute getAttribute() {
		return this.attribute;
	}

}