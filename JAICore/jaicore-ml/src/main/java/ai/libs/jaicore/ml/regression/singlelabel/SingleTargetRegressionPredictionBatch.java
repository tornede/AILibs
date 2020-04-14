package ai.libs.jaicore.ml.regression.singlelabel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.regression.evaluation.IRegressionPrediction;
import org.api4.java.ai.ml.regression.evaluation.IRegressionResultBatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SingleTargetRegressionPredictionBatch extends ArrayList<IRegressionPrediction> implements IRegressionResultBatch {

	private static final Logger LOGGER = LoggerFactory.getLogger(SingleTargetRegressionPredictionBatch.class);

	private static final long serialVersionUID = 1L;

	public SingleTargetRegressionPredictionBatch(final Collection<IRegressionPrediction> predictions) {
		this.addAll(predictions);
		LOGGER.debug("Predictions: {}", predictions.stream().map(i -> i.getDoublePrediction()).collect(Collectors.toList()));
	}

	@Override
	public int getNumPredictions() {
		return this.size();
	}

	@Override
	public List<? extends IRegressionPrediction> getPredictions() {
		return this;
	}

}