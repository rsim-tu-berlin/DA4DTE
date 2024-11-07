package gr.uoa.di.ai.gost;

import org.apache.jena.sparql.expr.*;
import org.apache.jena.arq.querybuilder.ExprFactory;
import org.apache.jena.sparql.syntax.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class GeoExprVisitor implements ExprVisitor{

    String type;
    List<LogicalBranch> branches;
    ElementGroup initialBlock;

    public GeoExprVisitor(){
        this.type = "INIT";
        this.branches = new ArrayList<>();
        this.initialBlock = null;
    }

    //Create the first branch of the query
    public void initializeBlock(ElementGroup elementGroup){
        this.initialBlock = elementGroup;
        LogicalBranch branch = new LogicalBranch(elementGroup);
        this.branches.add(branch);
    }

    @Override
    public void visit(ExprFunction0 exprFunction0) {
    }

    @Override
    public void visit(ExprFunction1 exprFunction1) {
        //Logical NOT in filter
        if(Objects.equals(exprFunction1.getOpName(), "!")){
            this.type="NOT";
            exprFunction1.getArg().visit(this);
        }
        else{
            if(Objects.equals(this.type, "NOT")) {
                ExprFactory factory = new ExprFactory();
                exprFunction1 = factory.not(exprFunction1);
            }
            this.branches.get(this.branches.size()-1).updateFilter(exprFunction1);
        }
    }

    @Override
    public void visit(ExprFunction2 exprFunction2) {

        //Logical OR in filter - create new branch
        if(Objects.equals(exprFunction2.getOpName(), "||")) {
            exprFunction2.getArg1().visit(this);
            this.type = "OR";
            LogicalBranch branch = new LogicalBranch(this.initialBlock);
            branches.add(branch);
            exprFunction2.getArg2().visit(this);
        }
        //Logical AND in filter - continue working on the current branch
        else if(Objects.equals(exprFunction2.getOpName(), "&&")) {
            exprFunction2.getArg1().visit(this);
            this.type = "AND";
            exprFunction2.getArg2().visit(this);
        }
        else{
            if(Objects.equals(this.type, "AND") || Objects.equals(this.type, "OR") || Objects.equals(this.type, "INIT") ){
                this.branches.get(this.branches.size()-1).updateFilter(exprFunction2);
            }
            this.type="NOP";
        }
    }

    @Override
    public void visit(ExprFunction3 exprFunction3) {
        this.branches.get(this.branches.size()-1).updateFilter(exprFunction3);
    }

    @Override
    public void visit(ExprFunctionN exprFunctionN) {
        this.branches.get(this.branches.size()-1).updateFilter(exprFunctionN,this.type);
    }

    @Override
    public void visit(ExprFunctionOp exprFunctionOp) {
    }

    @Override
    public void visit(NodeValue nodeValue) {
    }

    @Override
    public void visit(ExprVar exprVar) {
    }

    @Override
    public void visit(ExprAggregator exprAggregator) {
    }

    @Override
    public void visit(ExprNone exprNone) {
    }

    public String getType() {
        return type;
    }

    public Element mergeBranches(){

        //Only one branch return the block
        if(this.branches.size() == 1){
            return this.branches.get(0).process();
        }

        //Merge the branches to create the body of the materialized SPARQL query
        ElementUnion elementUnion = null;
        for(LogicalBranch branch:this.branches) {
            if (elementUnion == null)
                elementUnion = new ElementUnion(branch.process());
            else
                elementUnion.addElement(branch.process());
        }

        return elementUnion;
    }
}
