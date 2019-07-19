package ai.libs.jaicore.search.probleminputs.builders;

import org.api4.java.ai.graphsearch.problem.implicit.graphgenerator.PathGoalTester;
import org.api4.java.datastructure.graph.implicit.IGraphGenerator;

import ai.libs.jaicore.search.probleminputs.GraphSearchInput;

public abstract class SearchProblemInputBuilder<N, A, I extends GraphSearchInput<N, A>> {

	private IGraphGenerator<N, A> graphGenerator;
	private PathGoalTester<N, A> goalTester;

	public void setGraphGenerator(final IGraphGenerator<N, A> graphGenerator) {
		this.graphGenerator = graphGenerator;
	}

	public IGraphGenerator<N, A> getGraphGenerator() {
		return this.graphGenerator;
	}

	public void setGoalTester(final PathGoalTester<N, A> goalTester) {
		this.goalTester = goalTester;
	}

	public PathGoalTester<N, A> getGoalTester() {
		return this.goalTester;
	}

	public abstract I build();
}
