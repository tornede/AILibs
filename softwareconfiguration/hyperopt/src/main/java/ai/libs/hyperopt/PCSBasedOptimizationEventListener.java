package ai.libs.hyperopt;

import org.api4.java.algorithm.events.AlgorithmEvent;

import ai.libs.jaicore.graphvisualizer.events.graph.bus.AlgorithmEventListener;
import ai.libs.jaicore.graphvisualizer.events.graph.bus.HandleAlgorithmEventException;

public class PCSBasedOptimizationEventListener implements AlgorithmEventListener {

	@Override
	public void handleAlgorithmEvent(final AlgorithmEvent algorithmEvent) throws HandleAlgorithmEventException {
		PCSBasedOptimizationEvent event = (PCSBasedOptimizationEvent) algorithmEvent;

		System.out.println("event score: " + event.getScore());

	}

}