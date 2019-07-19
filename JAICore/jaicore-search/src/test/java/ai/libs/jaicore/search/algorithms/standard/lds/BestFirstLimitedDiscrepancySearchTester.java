package ai.libs.jaicore.search.algorithms.standard.lds;

import org.api4.java.ai.graphsearch.problem.IGraphSearch;

import ai.libs.jaicore.search.algorithms.GraphSearchSolutionIteratorTester;
import ai.libs.jaicore.search.probleminputs.GraphSearchInput;
import ai.libs.jaicore.search.probleminputs.GraphSearchWithNodeRecommenderInput;

public class BestFirstLimitedDiscrepancySearchTester extends GraphSearchSolutionIteratorTester {

	@Override
	public <N, A> IGraphSearch<?, ?, N, A> getSearchAlgorithm(final GraphSearchInput<N, A> problem) {
		GraphSearchWithNodeRecommenderInput<N, A> transformedProblem = new GraphSearchWithNodeRecommenderInput<>(problem, (n1, n2) -> 0);
		return new BestFirstLimitedDiscrepancySearch<>(transformedProblem);
	}

}
