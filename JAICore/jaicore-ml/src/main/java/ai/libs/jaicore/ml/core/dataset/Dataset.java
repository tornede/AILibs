package ai.libs.jaicore.ml.core.dataset;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.NoSuchElementException;

import org.api4.java.ai.ml.core.dataset.IDataset;
import org.api4.java.ai.ml.core.dataset.schema.ILabeledInstanceSchema;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.exception.DatasetCreationException;

public class Dataset implements ILabeledDataset<ILabeledInstance> {

	class DenseInstance implements ILabeledInstance {

		private int rowIndex;

		public DenseInstance(final int rowIndex) {
			this.rowIndex = rowIndex;
		}

		@Override
		public Double getAttributeValue(final int pos) {
			return Dataset.this.xMatrix.get(this.rowIndex)[pos];
		}

		@Override
		public int getNumAttributes() {
			return Dataset.this.xMatrix.get(this.rowIndex).length;
		}

		@Override
		public Object getLabel() {
			return Dataset.this.yMatrix.get(this.rowIndex);
		}

		@Override
		public double[] getPoint() {
			return Dataset.this.xMatrix.get(this.rowIndex);
		}

		@Override
		public double getPointValue(final int pos) {
			return Dataset.this.xMatrix.get(this.rowIndex)[pos];
		}

		@Override
		public ILabeledInstanceSchema getInstanceSchema() {
			return null;
		}

		@Override
		public Object[] getAttributes() {
			throw new UnsupportedOperationException();
		}
	}

	class InstanceIterator implements Iterator<ILabeledInstance>, ListIterator<ILabeledInstance> {

		private int nextIndex = 0;
		private int lastReturnedIndex = -1;

		private InstanceIterator() {
			// intentionally left blank
		}

		private InstanceIterator(final int startIndex) {
			this.nextIndex = startIndex;
		}

		@Override
		public boolean hasNext() {
			return this.nextIndex < Dataset.this.xMatrix.size();
		}

		@Override
		public ILabeledInstance next() {
			return new DenseInstance(this.nextIndex++);
		}

		@Override
		public ILabeledInstance previous() {
			return new DenseInstance((this.nextIndex--) - 2);
		}

		@Override
		public void add(final ILabeledInstance arg0) {
			Dataset.this.add(arg0);
		}

		@Override
		public boolean hasPrevious() {
			return this.nextIndex > 0;
		}

		@Override
		public int nextIndex() {
			return this.nextIndex;
		}

		@Override
		public int previousIndex() {
			return this.nextIndex - 2;
		}

		@Override
		public void set(final ILabeledInstance arg0) {
			Dataset.this.set(this.nextIndex - 1, arg0);
		}

		@Override
		public void remove() {
			if (this.lastReturnedIndex < 0) {
				throw new NoSuchElementException("No element to remove.");
			}
		}
	}

	private static final int DEFAULT_CAPACITY = 1;

	private List<double[]> xMatrix = new ArrayList<>();
	private List<Object> yMatrix = new ArrayList<>();

	private final ILabeledInstanceSchema instanceSchema;

	public Dataset(final ILabeledInstanceSchema instanceSchema) {
		this.instanceSchema = instanceSchema;
	}

	@Override
	public boolean add(final ILabeledInstance instance) {
		this.xMatrix.add(instance.getPoint());
		this.yMatrix.add(instance.getLabel());
		return true;
	}

	public void add(final double[] x, final Object y) {
		this.xMatrix.add(x);
		this.yMatrix.add(y);
	}

	public DenseInstance getInstance(final int index) {
		return new DenseInstance(index);
	}

	public boolean removeInstance(final int index) {
		if (index < 0 || index >= this.xMatrix.size()) {
			throw new NoSuchElementException("There is no such element to be removed. Invalid index (" + index + ") given.");
		}

		this.xMatrix.remove(index);
		this.yMatrix.remove(index);
		return true;
	}

	@Override
	public int size() {
		return this.xMatrix.size();
	}

	@Override
	public Iterator<ILabeledInstance> iterator() {
		return new InstanceIterator();
	}

	@Override
	public void add(final int index, final ILabeledInstance element) {
		this.xMatrix.add(index, element.getPoint());
		this.yMatrix.add(index, element.getLabel());
	}

	@Override
	public boolean addAll(final Collection<? extends ILabeledInstance> c) {
		c.stream().forEach(this::add);
		return true;
	}

	@Override
	public void clear() {
		this.xMatrix.clear();
		this.yMatrix.clear();
	}

	@Override
	public ILabeledInstanceSchema getInstanceSchema() {
		return this.instanceSchema;
	}

	@Override
	public IDataset<ILabeledInstance> createEmptyCopy() throws DatasetCreationException, InterruptedException {
		return new Dataset(this.instanceSchema);
	}

	@Override
	public ILabeledInstance get(final int pos) {
		return new ILabeledInstance() {

			@Override
			public ILabeledInstanceSchema getInstanceSchema() {
				return Dataset.this.instanceSchema;
			}

			@Override
			public double getPointValue(final int pos2) {
				return Dataset.this.xMatrix.get(pos)[pos2];
			}

			@Override
			public double[] getPoint() {
				return Dataset.this.xMatrix.get(pos);
			}

			@Override
			public Object[] getAttributes() {
				// TODO Auto-generated method stub
				return null;
			}

			@Override
			public Object getAttributeValue(final int pos) {
				return null;
			}

			@Override
			public Object getLabel() {
				return Dataset.this.yMatrix.get(pos);
			}
		};
	}

	@Override
	public Object[][] getFeatureMatrix() {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public boolean isEmpty() {
		return this.xMatrix.isEmpty();
	}

	@Override
	public boolean contains(final Object o) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public Object[] toArray() {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public <T> T[] toArray(final T[] a) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public boolean remove(final Object o) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public boolean containsAll(final Collection<?> c) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public boolean addAll(final int index, final Collection<? extends ILabeledInstance> c) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public boolean removeAll(final Collection<?> c) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public boolean retainAll(final Collection<?> c) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public ILabeledInstance set(final int index, final ILabeledInstance element) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public ILabeledInstance remove(final int index) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public int indexOf(final Object o) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public int lastIndexOf(final Object o) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public ListIterator<ILabeledInstance> listIterator() {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public ListIterator<ILabeledInstance> listIterator(final int index) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public List<ILabeledInstance> subList(final int fromIndex, final int toIndex) {
		throw new UnsupportedOperationException("not implemented!");
	}

	@Override
	public Object[] getLabelVector() {
		throw new UnsupportedOperationException("not implemented!");
	}

}
