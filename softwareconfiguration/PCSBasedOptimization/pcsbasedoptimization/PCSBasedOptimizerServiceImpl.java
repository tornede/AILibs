package ai.libs.hasco.pcsbasedoptimization;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.Parameter;
import ai.libs.hasco.pcsbasedoptimization.proto.PCSBasedComponentProto;
import ai.libs.hasco.pcsbasedoptimization.proto.PCSBasedEvaluationResponseProto;
import ai.libs.hasco.pcsbasedoptimization.proto.PCSBasedOptimizerServiceGrpc.PCSBasedOptimizerServiceImplBase;
import ai.libs.hasco.pcsbasedoptimization.proto.PCSBasedParameterProto;
import ai.libs.jaicore.basic.IObjectEvaluator;
import ai.libs.jaicore.basic.algorithm.exceptions.ObjectEvaluationFailedException;
import io.grpc.stub.StreamObserver;

/**
 * gRPC service implementation for PCSBasedOptimizers
 * 
 * @author kadirayk
 *
 */
public class PCSBasedOptimizerServiceImpl extends PCSBasedOptimizerServiceImplBase {

	private PCSBasedOptimizerInput input;
	private IObjectEvaluator<ComponentInstance, Double> evaluator;

	public PCSBasedOptimizerServiceImpl(IObjectEvaluator<ComponentInstance, Double> evaluator,
			PCSBasedOptimizerInput input) {
		this.evaluator = evaluator;
		this.input = input;
	}

	/**
	 * Optimizer scripts call this method with a component name and a list of
	 * parameters for that component. The corresponding component will be
	 * instantiated with the parameters as a componentInstance.
	 * 
	 * ComponentInstance will be evaluated with the given evaluator. A response
	 * containing an evalutation score will return to the caller script
	 */
	@Override
	public void evaluate(PCSBasedComponentProto request,
			StreamObserver<PCSBasedEvaluationResponseProto> responseObserver) {
		Collection<Component> components = input.getComponents();
		ComponentInstance componentInstance = resolveSatisfyingInterfaces(components, request.getName(),
				request.getParametersList());

		Double score = 0.0;
		try {
			score = evaluator.evaluate(componentInstance);
		} catch (InterruptedException | ObjectEvaluationFailedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		PCSBasedEvaluationResponseProto response = PCSBasedEvaluationResponseProto.newBuilder().setResult(score)
				.build();

		responseObserver.onNext(response);
		responseObserver.onCompleted();
	}

	/**
	 * Creates a ComponentInstance based on the given componentName, components, and
	 * a list of component parameters. Recursively resolves satisfying interfaces.
	 * 
	 * @param components
	 * @param componentName
	 * @param params
	 * @return
	 */
	private ComponentInstance resolveSatisfyingInterfaces(Collection<Component> components, String componentName,
			List<PCSBasedParameterProto> params) {
		Component cmp = components.stream().filter(c -> c.getName().contains(componentName)).findFirst().get();
		Set<String> requiredInterfaces = cmp.getRequiredInterfaces().keySet();

		Set<Parameter> hascoParams = cmp.getParameters();

		Map<String, String> componentParameters = new HashMap<>();

		for (Parameter hp : hascoParams) {
			Optional<PCSBasedParameterProto> op = params.stream()
					.filter(p -> p.getKey().contains(componentName + "." + hp)).findFirst();
			if (op.isPresent()) {
				PCSBasedParameterProto param = op.get();
				int indexOfLastDot = param.getKey().lastIndexOf(".");
				String key = param.getKey().substring(indexOfLastDot + 1);
				componentParameters.put(key, param.getValue());
			}
		}

		if (requiredInterfaces == null || requiredInterfaces.isEmpty()) {
			Map<String, ComponentInstance> satisfyingInterfaces = new HashMap<>();
			return new ComponentInstance(cmp, componentParameters, satisfyingInterfaces);
		}

		for (String requiredInterface : requiredInterfaces) {
			PCSBasedParameterProto param = params.stream().filter(p -> p.getKey().equals(requiredInterface)).findFirst()
					.get();
			String satisfyingClassName = param.getValue();
			ComponentInstance compInstance = resolveSatisfyingInterfaces(components, satisfyingClassName, params);
			Map<String, ComponentInstance> satisfyingInterfaces = new HashMap<>();
			satisfyingInterfaces.put(componentName, compInstance);
			return new ComponentInstance(cmp, componentParameters, satisfyingInterfaces);
		}
		return null;
	}

}