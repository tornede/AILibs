package ai.libs.jaicore.ml.core.dataset.schema;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.api4.java.ai.ml.core.dataset.schema.IInstanceSchema;
import org.api4.java.ai.ml.core.dataset.schema.attribute.IAttribute;

public class InstanceSchema implements IInstanceSchema {

	private String relationName;
	private final List<IAttribute> attributeList;

	public InstanceSchema(final String relationName, final Collection<IAttribute> attributeList) {
		this.relationName = relationName;
		this.attributeList = new ArrayList<>(attributeList);
	}

	@Override
	public IAttribute getAttribute(final int pos) {
		return this.attributeList.get(pos);
	}

	@Override
	public int getNumAttributes() {
		return this.attributeList.size();
	}

	@Override
	public String getRelationName() {
		return this.relationName;
	}

}
