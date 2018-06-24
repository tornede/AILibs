package jaicore.graphvisualizer.events.add;

public class InfoEvent {
    private int maxIndex;
    private long maxTime;
    private int numberOfDataSupplier;

    public InfoEvent(int maxIndex, long maxTime, int numberOfDataSupplier ){
        this.maxIndex = maxIndex;
        this.maxTime = maxTime;
        this.numberOfDataSupplier = numberOfDataSupplier;

    }

    public int getMaxIndex() {
        return maxIndex;
    }

    public long getMaxTime() {
        return maxTime;
    }

    public int getNumberOfDataSupplier(){return numberOfDataSupplier;}
}
