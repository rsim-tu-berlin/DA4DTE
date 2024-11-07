package gr.uoa.di.ai.gost.aggregators;

import org.apache.jena.sparql.expr.aggregate.Accumulator;
import org.apache.jena.sparql.expr.aggregate.AccumulatorFactory;
import org.apache.jena.sparql.expr.aggregate.AggCustom;

public class AreaAggFactory implements AccumulatorFactory {
    @Override
    public Accumulator createAccumulator(AggCustom aggCustom, boolean b) {
        return new AreaAccumulator();
    }
}
