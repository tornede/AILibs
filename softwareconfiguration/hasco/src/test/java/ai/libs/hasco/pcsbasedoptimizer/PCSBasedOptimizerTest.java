package ai.libs.hasco.pcsbasedoptimizer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.pcsbasedoptimization.BOHBOptimizer;
import ai.libs.hasco.pcsbasedoptimization.ComponentInstanceEvaluator;
import ai.libs.hasco.pcsbasedoptimization.HASCOToPCSConverter;
import ai.libs.hasco.pcsbasedoptimization.HyperBandOptimizer;
import ai.libs.hasco.pcsbasedoptimization.OptimizationException;
import ai.libs.hasco.pcsbasedoptimization.PCSBasedOptimizerInput;
import ai.libs.hasco.pcsbasedoptimization.SMACOptimizer;
import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.basic.IObjectEvaluator;
import ai.libs.mlplan.multiclass.wekamlplan.weka.WekaPipelineFactory;

/**
 * 
 * @author kadirayk
 *
 */
public class PCSBasedOptimizerTest {

	private static final File HASCOFileInput = new File("../mlplan/resources/automl/searchmodels/weka/autoweka.json");

	PCSBasedOptimizerInput input;
	IObjectEvaluator<ComponentInstance, Double> evaluator;

	@Before
	public void init() {
		ComponentLoader cl = null;
		try {
			cl = new ComponentLoader(HASCOFileInput);
		} catch (IOException e) {

		}
		Collection<Component> components = cl.getComponents();
		String requestedInterface = "BaseClassifier";
		input = new PCSBasedOptimizerInput(components, requestedInterface);
		WekaPipelineFactory classifierFactory = new WekaPipelineFactory();
		evaluator = new ComponentInstanceEvaluator(classifierFactory, "testrsc/iris.arff");
	}

	@Ignore
	@Test
	public void conversionTest() throws Exception {
		HASCOToPCSConverter.generatePCSFile(input, "output/");
		File pcsFile = new File("output/StackingEstimator.pcs");
		assertTrue(pcsFile.exists());
		String content = FileUtil.readFileAsString(pcsFile);
		assertTrue(content.contains(
				"estimator {sklearn.naive_bayes.GaussianNB,sklearn.naive_bayes.BernoulliNB,sklearn.naive_bayes.MultinomialNB,sklearn.tree.DecisionTreeClassifier,sklearn.ensemble.RandomForestClassifier,sklearn.ensemble.GradientBoostingClassifier,sklearn.neighbors.KNeighborsClassifier,sklearn.svm.LinearSVC}"));
		assertTrue(content
				.contains("sklearn.naive_bayes.BernoulliNB.fit_prior|estimator in {sklearn.naive_bayes.BernoulliNB}"));
	}

	@Ignore
	@Test
	public void formatTest() throws Exception {
		HASCOToPCSConverter.generatePCSFile(input, "output/");
		File pcsFile = new File("output/StackingEstimator.pcs");
		List<String> content = FileUtil.readFileAsList(pcsFile);
		for (String line : content) {
			if (line.contains("Conditionals:")) {
				return;
			}
			if (line.indexOf("{") != -1) { // categorical
				int curlyOpen = line.indexOf("{");
				assertEquals(line.charAt(curlyOpen - 1), " ".toCharArray()[0]); // there must be a space before opening
																				// curly braces
				int curlyClose = line.indexOf("}");
				assertNotEquals(-1, curlyClose); // there must be a closing curly brace
				assertTrue(curlyClose > curlyOpen); // closing must come after opening
				int squareOpen = line.indexOf("[");
				int squareClose = line.indexOf("]");
				assertNotEquals(-1, squareOpen); // each line should have an opening square bracket
				assertNotEquals(-1, squareClose); // and a closing one
				assertTrue(squareClose > squareOpen); // closing must be after opening
			}
		}

	}

	@Ignore
	@Test(expected = OptimizationException.class)
	public void SMACOptimizationExceptionTest() throws Exception {
		HASCOToPCSConverter.generatePCSFile(input, "PCSBasedOptimizerScripts/SMACOptimizer/");
		SMACOptimizer smacOptimizer = SMACOptimizer.SMACOptimizerBuilder(input, evaluator).executionPath("wrongPath")
				.algoRunsTimelimit(99).runCountLimit(11).build();
		smacOptimizer.optimize("weka.classifiers.functions.Logistic");
	}

	@Ignore
	@Test(expected = OptimizationException.class)
	public void HyperBandOptimizationExceptionTest() throws Exception {
		HASCOToPCSConverter.generatePCSFile(input, "PCSBasedOptimizerScripts/HyperBandOptimizer/");
		HyperBandOptimizer optimizer = HyperBandOptimizer.HyperBandOptimizerBuilder(input, evaluator)
				.executionPath("wrongPath").maxBudget(230.0).minBudget(9.0).nIterations(4).build();
		optimizer.optimize("weka.classifiers.functions.Logistic");

	}

	@Ignore
	@Test(expected = OptimizationException.class)
	public void BOHBOptimizationExceptionTest() throws Exception {
		HASCOToPCSConverter.generatePCSFile(input, "PCSBasedOptimizerScripts/BOHBOptimizer/");
		BOHBOptimizer optimizer = BOHBOptimizer.BOHBOptimizerBuilder(input, evaluator).executionPath("wrongPath")
				.maxBudget(230.0).minBudget(9.0).nIterations(4).build();
		optimizer.optimize("weka.classifiers.functions.Logistic");

	}

	@Test
	public void spawnSMACTest() throws Exception {
		HASCOToPCSConverter.generatePCSFile(input, "PCSBasedOptimizerScripts/SMACOptimizer/");
		SMACOptimizer smacOptimizer = SMACOptimizer.SMACOptimizerBuilder(input, evaluator)
				.executionPath("PCSBasedOptimizerScripts/SMACOptimizer").algoRunsTimelimit(99).runCountLimit(11)
				.build();
		smacOptimizer.optimize("weka.classifiers.functions.Logistic");
	}

	@Ignore
	@Test
	public void spawnHyperBandTest() throws Exception {
		HASCOToPCSConverter.generatePCSFile(input, "PCSBasedOptimizerScripts/HyperBandOptimizer/");
		HyperBandOptimizer optimizer = HyperBandOptimizer.HyperBandOptimizerBuilder(input, evaluator)
				.executionPath("PCSBasedOptimizerScripts/HyperBandOptimizer").maxBudget(230.0).minBudget(9.0)
				.nIterations(4).build();
		optimizer.optimize("weka.classifiers.functions.Logistic");

	}

	@Ignore
	@Test
	public void spawnBOHBTest() throws Exception {
		HASCOToPCSConverter.generatePCSFile(input, "PCSBasedOptimizerScripts/BOHBOptimizer/");
		BOHBOptimizer optimizer = BOHBOptimizer.BOHBOptimizerBuilder(input, evaluator)
				.executionPath("PCSBasedOptimizerScripts/BOHBOptimizer").maxBudget(230.0).minBudget(9.0).nIterations(4)
				.build();
		optimizer.optimize("weka.classifiers.functions.Logistic");

	}

}
