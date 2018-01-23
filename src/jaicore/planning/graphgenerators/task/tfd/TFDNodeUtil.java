package jaicore.planning.graphgenerators.task.tfd;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import jaicore.logic.fol.structure.CNFFormula;
import jaicore.logic.fol.structure.Clause;
import jaicore.logic.fol.structure.ConstantParam;
import jaicore.logic.fol.structure.Literal;
import jaicore.logic.fol.structure.Monom;
import jaicore.planning.graphgenerators.task.TaskPlannerUtil;
import jaicore.planning.graphgenerators.task.ceociptfd.EvaluablePredicate;
import jaicore.planning.model.conditional.CEAction;
import jaicore.planning.model.conditional.CEOperation;
import jaicore.planning.model.core.Action;
import jaicore.planning.model.strips.StripsAction;
import jaicore.planning.model.strips.StripsOperation;
import jaicore.planning.model.task.stn.MethodInstance;

public class TFDNodeUtil {

	private static Map<List<TFDNode>, Integer> cache = new HashMap<>();
	private final TaskPlannerUtil util;
	private final Map<String, EvaluablePredicate> evaluablePlanningPredicates;
	
	public TFDNodeUtil(Map<String, EvaluablePredicate> evaluablePlanningPredicates) {
		super();
		this.evaluablePlanningPredicates = evaluablePlanningPredicates;
		util = new TaskPlannerUtil(evaluablePlanningPredicates);
	}

	private boolean checkDoubleRestProblemComputationOccurrence(List<TFDNode> path) {
		if (cache.containsKey(path)) {
			System.out.println("already seen path " + cache.get(path) + " times");
			return false;
		}
		cache.put(path, 0);
		return true;
	}
	
	public List<TFDNode> getPathOfNode(TFDNode node, Map<TFDNode, TFDNode> parentMap) {
		
		/* compute path for node */
		List<TFDNode> path = new ArrayList<>();
		TFDNode current = node;
		while (current != null) {
			path.add(current);
			current = parentMap.get(current);
		}
		Collections.reverse(path);
		return path;
	}

	public TFDRestProblem getRestProblem(List<TFDNode> path) {

		/* get last node in list with explicit rest problem formulation */
		assert checkDoubleRestProblemComputationOccurrence(path) : "We must not generate the information of a node twice!";

		/* identify latest node that has an explicit rest problem attached */
		TFDNode latest = null;
		for (TFDNode n : path) {
			if (n.getProblem() != null)
				latest = n;
		}

		/* set iterator to the last check point node */
		Iterator<TFDNode> i = path.iterator();
		TFDNode init = null;
		do {
			TFDNode n = i.next();
			if (n == latest)
				init = n;
		} while (init == null);

		/* compute the rest problem going from there */
		Monom state = new Monom(init.getState(), false);
		List<Literal> remainingTasks = new ArrayList<>(init.getRemainingTasks());
		while (i.hasNext()) {
			TFDNode n = i.next();

			/* compute updated state */
			Action appliedAction = n.getAppliedAction();
			if (appliedAction != null) {
				updateState(state, appliedAction);
			}

			/* compute remaining tasks */
			remainingTasks.remove(0);
			MethodInstance appliedMethodInstance = n.getAppliedMethodInstance();
			if (appliedMethodInstance != null) {
				int j = 0;
				for (Literal remainingTask : util.getTaskChainOfTotallyOrderedNetwork(appliedMethodInstance.getNetwork())) {
					remainingTasks.add(j++, remainingTask);
				}
			}
		}
		return new TFDRestProblem(state, new ArrayList<>(remainingTasks));
	}

	public Monom getState(List<TFDNode> path) {
		return getRestProblem(path).getState();
	}

	public void updateState(Monom state, Action appliedAction) {

		// assert state.containsAll(appliedAction.getPrecondition().stream().filter(lit -> lit.isPositive()).collect(Collectors.toList())) && SetUtil.disjoint(state,
		// appliedAction.getPrecondition().stream().filter(lit -> lit.isNegated()).collect(Collectors.toList())) : ("Action " + appliedAction + " is supposed to be aplpicable in state " + state + "
		// but it is not!");

		/* apply effects of action (STRIPS) */
		if (appliedAction.getOperation() instanceof StripsOperation) {
			Action a = new StripsAction((StripsOperation) appliedAction.getOperation(), appliedAction.getGrounding());
			state.removeAll(((StripsAction) a).getDeleteList());
			state.addAll(((StripsAction) a).getAddList());
		}

		/* apply effects of action (ConditionalEffect operations) */
		else if (appliedAction.getOperation() instanceof CEOperation) {
			CEAction a = new CEAction((CEOperation) appliedAction.getOperation(), appliedAction.getGrounding());
			Map<CNFFormula, Monom> addLists = a.getAddLists();
			for (CNFFormula condition : addLists.keySet()) {
				
				/* evaluate interpreted predicates */
				CNFFormula modifiedCondition = new CNFFormula();
				boolean conditionIsSatisfiable = true;
				for (Clause c : condition) {
					Clause modifiedClause = new Clause();
					boolean clauseContainsTrue = false;
					for (Literal l : c) {
						if (l.getPropertyName().startsWith("$")) {
							EvaluablePredicate predicate = util.getEvaluablePlanningPredicates().get(l.getPropertyName().substring(1));
							if (predicate == null)
								throw new IllegalArgumentException("Action has evaluable predicate " + l.getPropertyName() + " in its conditional postcondition, but this predicate is not defined in the evaluable predicates!");
							boolean testResult = predicate.test(state, l.getConstantParams().toArray(new ConstantParam[0]));
							if (testResult == l.isPositive()) {
								clauseContainsTrue = true;
								break;
							}
							else;
								// simply ignore this predicate
						}
						else
							modifiedClause.add(l);
						
						/* if the clause is not empty, add it to the condition */
						if (!clauseContainsTrue) {
							if (!modifiedClause.isEmpty())
								modifiedCondition.add(modifiedClause);
							else {
								conditionIsSatisfiable = false;
								break;
							}
						}
					}
				}
				if (conditionIsSatisfiable && modifiedCondition.entailedBy(state)) {
					state.addAll(addLists.get(condition));
				}
			}
			Map<CNFFormula, Monom> deleteLists = a.getDeleteLists();
			for (CNFFormula condition : deleteLists.keySet()) {
				if (condition.entailedBy(state)) {
					state.removeAll(deleteLists.get(condition));
				}
			}
		} else {
			System.err.println("No support for operations of class " + appliedAction.getOperation().getClass());
		}
	}

	@SuppressWarnings("unused")
	private boolean checkConsistency(Monom state, Map<CNFFormula, Monom> addLists) {
		for (Literal lit : state) {
			if (lit.getPropertyName().equals("cluster")) {
				String clusterName = lit.getConstantParams().get(0).getName();
				boolean foundSmallest = false;
				boolean foundRepresentant = false;
				for (Literal lit2 : state) {
					if (lit2.getPropertyName().equals("smallest") && lit2.getConstantParams().get(1).getName().equals(clusterName)) {
						foundSmallest = true;
						String smallestItem = lit2.getConstantParams().get(0).getName();
						List<ConstantParam> params = new ArrayList<>();
						params.add(new ConstantParam(smallestItem));
						params.add(new ConstantParam(clusterName));
						Literal lit3 = new Literal("in", params);
						if (!state.contains(lit3)) {
							throw new IllegalStateException(
									"Smallest item in cluster " + clusterName + " is " + smallestItem + ", which is not even contained according to state " + state + "!");
						}
						for (Literal lit4 : state) {
							if (lit4.getPropertyName().equals("in") && lit4.getConstantParams().get(1).getName().equals(clusterName)) {
								if (state.contains(new Literal("bigger('" + smallestItem + "','" + lit4.getConstantParams().get(0).getName() + "')"))) {
									throw new IllegalStateException("Cluster " + clusterName + " has " + smallestItem + " as smallest item, but "
											+ lit4.getConstantParams().get(0).getName() + " is smaller");
								}
							}
						}
						break;
					}
					if (lit2.getPropertyName().equals("represents") && lit2.getConstantParams().get(1).getName().equals(clusterName)) {
						foundRepresentant = true;
						break;
					}
				}
				if (!foundSmallest && !foundRepresentant) {
					// for (CNFFormula condition : addLists.keySet()) {
					// CNFFormula evaledCondition = LogicUtil.evalEqualityLiteralsUnderUNA(condition);
					// System.out.println(evaledCondition + ": " + (evaledCondition.isConsistent() && state.containsAll(evaledCondition)));
					// if (evaledCondition.isConsistent())
					// System.out.println(state);
					// }
					// System.err.println(logger.toString());
					throw new IllegalStateException("State " + state + " does not specify a smallest element for cluster " + clusterName + " after applying addList " + addLists);
				}
			}
		}
		return true;
	}

	public List<Literal> getRemainingTasks(List<TFDNode> path) {
		return getRestProblem(path).getRemainingTasks();
	}
}