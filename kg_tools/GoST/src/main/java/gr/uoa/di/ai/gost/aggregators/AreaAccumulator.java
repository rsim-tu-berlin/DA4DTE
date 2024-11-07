package gr.uoa.di.ai.gost.aggregators;

import org.apache.jena.sparql.engine.binding.Binding;
import org.apache.jena.sparql.expr.NodeValue;
import org.apache.jena.sparql.expr.aggregate.Accumulator;
import org.apache.jena.sparql.function.FunctionEnv;

public class AreaAccumulator implements Accumulator {
    @Override
    public void accumulate(Binding binding, FunctionEnv functionEnv) {

    }

    @Override
    public NodeValue getValue() {
        return null;
    }
}
