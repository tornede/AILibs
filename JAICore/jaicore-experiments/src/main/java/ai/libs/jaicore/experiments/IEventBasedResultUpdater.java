package ai.libs.jaicore.experiments;

import java.util.Map;

import org.api4.java.algorithm.events.IAlgorithmEvent;

public interface IEventBasedResultUpdater {
	public void processEvent(IAlgorithmEvent e, Map<String, Object> currentResults);

	public void finish(Map<String, Object> currentResults);
}
