package ai.libs.jaicore.search.model.travesaltree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.api4.java.ai.graphsearch.problem.pathsearch.pathevaluation.IEvaluatedPath;
import org.api4.java.datastructure.graph.IPath;

import ai.libs.jaicore.logging.ToJSONStringUtil;

public class BackPointerPath<N, A, V extends Comparable<V>> implements IEvaluatedPath<N, A, V> {
	private final N nodeLabel;
	private final A edgeLabelToParent;
	private boolean goal;
	protected BackPointerPath<N, A, V> parent;
	private final Map<String, Object> annotations = new HashMap<>(); // for nodes effectively examined

	public BackPointerPath(final BackPointerPath<N, A, V> parent, final N point, final A edgeLabelToParent) {
		super();
		this.parent = parent;
		this.nodeLabel = point;
		this.edgeLabelToParent = edgeLabelToParent;
	}

	public BackPointerPath<N, A, V> getParent() {
		return this.parent;
	}

	@Override
	public N getHead() {
		return this.nodeLabel;
	}

	@SuppressWarnings("unchecked")
	@Override
	public V getScore() {
		return (V) this.annotations.get("f");
	}

	public void setParent(final BackPointerPath<N, A, V> newParent) {
		this.parent = newParent;
	}

	public void setScore(final V internalLabel) {
		this.setAnnotation("f", internalLabel);
	}

	public void setAnnotation(final String annotationName, final Object annotationValue) {
		this.annotations.put(annotationName, annotationValue);
	}

	public Object getAnnotation(final String annotationName) {
		return this.annotations.get(annotationName);
	}

	public Map<String, Object> getAnnotations() {
		return this.annotations;
	}

	public boolean isGoal() {
		return this.goal;
	}

	public void setGoal(final boolean goal) {
		this.goal = goal;
	}

	public List<BackPointerPath<N, A, V>> path() {
		List<BackPointerPath<N, A, V>> path = new ArrayList<>();
		BackPointerPath<N, A, V> current = this;
		while (current != null) {
			path.add(0, current);
			current = current.parent;
		}
		return path;
	}

	@Override
	public List<N> getNodes() {
		List<N> path = new ArrayList<>();
		BackPointerPath<N, A, V> current = this;
		while (current != null) {
			path.add(0, current.nodeLabel);
			current = current.parent;
		}
		return path;
	}

	public String getString() {
		String s = "Node [ref=";
		s += this.toString();
		s += ", externalLabel=";
		s += this.nodeLabel;
		s += ", goal";
		s += this.goal;
		s += ", parentRef=";
		if (this.parent != null) {
			s += this.parent.toString();
		} else {
			s += "null";
		}
		s += ", annotations=";
		s += this.annotations;
		s += "]";
		return s;
	}

	@Override
	public String toString() {
		Map<String, Object> fields = new HashMap<>();
		fields.put("externalLabel", this.nodeLabel);
		fields.put("goal", this.goal);
		fields.put("annotations", this.annotations);
		return ToJSONStringUtil.toJSONString(this.getClass().getSimpleName(), fields);
	}

	@Override
	public List<A> getArcs() {
		if (this.parent == null) {
			return new LinkedList<>();
		}
		List<A> pathToHere = this.parent.getArcs();
		pathToHere.add(this.edgeLabelToParent);
		return pathToHere;
	}

	public A getEdgeLabelToParent() {
		return this.edgeLabelToParent;
	}

	@Override
	public N getRoot() {
		return this.parent == null ? this.nodeLabel : this.parent.getRoot();
	}

	@Override
	public IPath<N, A> getPathToParentOfHead() {
		return this.parent;
	}
}