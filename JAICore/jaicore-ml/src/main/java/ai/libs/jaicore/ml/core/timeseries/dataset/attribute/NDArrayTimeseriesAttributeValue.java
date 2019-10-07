package ai.libs.jaicore.ml.core.timeseries.dataset.attribute;

import org.api4.java.ai.ml.core.dataset.schema.attribute.ITimeseriesAttribute;
import org.api4.java.ai.ml.core.dataset.schema.attribute.ITimeseriesAttributeValue;
import org.api4.java.common.timeseries.ITimeseries;
import org.nd4j.linalg.api.ndarray.INDArray;

public class NDArrayTimeseriesAttributeValue implements ITimeseriesAttributeValue<INDArray> {

	private ITimeseriesAttribute attribute;
	private ITimeseries<INDArray> value;

	public NDArrayTimeseriesAttributeValue(final ITimeseriesAttribute attribute, final ITimeseries<INDArray> value) {
		this.value = value;
	}

	@Override
	public ITimeseries<INDArray> getValue() {
		return this.value;
	}

	@Override
	public ITimeseriesAttribute getAttribute() {
		return this.attribute;
	}

}
