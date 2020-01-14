package ai.libs.hasco.pcsbasedoptimization;

import java.util.Collection;

import ai.libs.hasco.model.Component;

/**
 * 
 * @author kadirayk
 *
 */
public class PCSBasedOptimizerInput {

	private Collection<Component> components;

	private String requestedInterface;
	private String requestedComponent;

	public PCSBasedOptimizerInput(Collection<Component> components, String requestedComponent, String requestedInterface) {
		this.components = components;
		this.requestedInterface = requestedInterface;
		this.requestedComponent = requestedComponent;
	}

	public Collection<Component> getComponents() {
		return components;
	}


	public String getRequestedInterface() {
		return requestedInterface;
	}
	
	public String getRequestedComponent() {
		return requestedComponent;
	}


}
