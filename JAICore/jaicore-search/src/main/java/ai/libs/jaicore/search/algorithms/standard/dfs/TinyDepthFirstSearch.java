package ai.libs.jaicore.search.algorithms.standard.dfs;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

import org.api4.java.ai.graphsearch.problem.implicit.graphgenerator.NodeGoalTester;
import org.api4.java.datastructure.graph.implicit.NodeExpansionDescription;
import org.api4.java.datastructure.graph.implicit.SingleRootGenerator;
import org.api4.java.datastructure.graph.implicit.SuccessorGenerator;

import ai.libs.jaicore.search.model.other.SearchGraphPath;
import ai.libs.jaicore.search.probleminputs.GraphSearchInput;

public class TinyDepthFirstSearch<N, A> {
	private final List<SearchGraphPath<N, A>> solutionPaths = new LinkedList<>();
	private final SuccessorGenerator<N, A> successorGenerator;
	private final NodeGoalTester<N, A> goalTester;
	private final N root;
	private final Deque<N> path = new LinkedList<>();

	public TinyDepthFirstSearch(final GraphSearchInput<N, A> problem) {
		super();
		this.root = ((SingleRootGenerator<N>) problem.getGraphGenerator().getRootGenerator()).getRoot();
		this.goalTester = (NodeGoalTester<N, A>) problem.getGoalTester();
		this.successorGenerator = problem.getGraphGenerator().getSuccessorGenerator();
		this.path.add(this.root);
	}

	public void run() throws InterruptedException {
		this.dfs(this.root);
	}

	public void dfs(final N head) throws InterruptedException {
		if (this.goalTester.isGoal(head)) {
			this.solutionPaths.add(new SearchGraphPath<>(new ArrayList<>(this.path)));
		}
		else {

			/* expand node and invoke dfs for each child in order */
			List<NodeExpansionDescription<N,A>> successors = this.successorGenerator.generateSuccessors(head);
			for (NodeExpansionDescription<N,A> succ : successors) {
				N to = succ.getTo();
				this.path.addFirst(to);
				this.dfs(to);
				N removed = this.path.removeFirst();
				assert removed == to : "Expected " + to + " but removed " + removed;
			}
		}
	}

	public List<SearchGraphPath<N, A>> getSolutionPaths() {
		return this.solutionPaths;
	}
}
