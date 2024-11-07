package gr.uoa.di.ai.gost;

import gr.uoa.di.ai.gost.exceptions.NoSuchGeoExpressionException;
import org.apache.jena.arq.querybuilder.ExprFactory;
import org.apache.jena.graph.Node;
import org.apache.jena.graph.NodeFactory;
import org.apache.jena.graph.Triple;
import org.apache.jena.sparql.expr.*;
import org.apache.jena.sparql.syntax.*;

import java.util.ArrayList;
import java.util.List;

public class LogicalBranch {
    ElementGroup elementGroup;
    ElementFilter elementFilter;
    List<ElementTriplesBlock> materializedElements;

    public LogicalBranch(ElementGroup elementGroup){
        this.elementGroup = new ElementGroup();
        for(Element e: elementGroup.getElements()){
            this.elementGroup.addElement(e);
        }
        this.materializedElements = new ArrayList<>();
        this.elementFilter = null;
    }

    public void updateFilter(Expr expr){
        if(expr instanceof ExprFunction1 || expr instanceof ExprFunction2 || expr instanceof ExprFunction3 || expr instanceof ExprFunctionN) {
            if (this.elementFilter == null) {
                this.elementFilter = new ElementFilter(expr);
            } else {
                ExprFactory factory = new ExprFactory();
                Expr andExr = factory.and(expr, this.elementFilter.getExpr());
                this.elementFilter = new ElementFilter(andExr);
            }
        }
    }

    public void updateFilter(ExprFunctionN exprFunctionN, String type){
        String materialized = GeoDictionary.getMaterialized(exprFunctionN.getFunctionIRI());
        //GeoSPARQL topological function
        if(materialized!=null){
            Node subject = generateMaterializedNode(exprFunctionN.getArg(1));
            Node object = generateMaterializedNode(exprFunctionN.getArg(2));
            ElementTriplesBlock elementTriplesBlock = new ElementTriplesBlock();
            elementTriplesBlock.addTriple(
                    Triple.create(
                            subject,
                            NodeFactory.createURI(materialized),
                            object
                    )
            );

            if(type.equals("NOT")) {
                ElementGroup minusGroup = new ElementGroup();
                for(Element e:this.elementGroup.getElements()) minusGroup.addElement(e);
                minusGroup.addElement(elementTriplesBlock);
                elementGroup.addElement(new ElementMinus(minusGroup));
            }
            else{
                materializedElements.add(elementTriplesBlock);
                //enhance the sparql block with the materialized element
                elementGroup.addElement(elementTriplesBlock);
            }
        }
        else{
            updateFilter(exprFunctionN);
        }
    }

    private Node generateMaterializedNode(Expr expr){
        String geoExpr=null;
        if(expr instanceof ExprFunctionN){
            ExprFunctionN exprFunctionN = (ExprFunctionN) expr;
            if(exprFunctionN.getFunctionIRI().equals("http://strdf.di.uoa.gr/ontology#transform") ||
                exprFunctionN.getFunctionIRI().equals("http://strdf.di.uoa.gr/ontology#buffer")) {
                geoExpr = exprFunctionN.getArg(1).toString();
            }
            else
                throw new UnsupportedOperationException("GeoSpatial transformation function not recognized");
        }
        else{
            geoExpr = expr.toString();
        }

        Node node = null;
        try {
            node = processNodeName(geoExpr);
        } catch (NoSuchGeoExpressionException e) {
            e.printStackTrace();
            System.exit(1);
        }

        return node;
    }

    private Node processNodeName(String expr) throws NoSuchGeoExpressionException {
        String name = GeoDictionary.getMapping(expr);
        if(name==null)
            throw new NoSuchGeoExpressionException(expr + " is not a geospatial expression");

        if(name.startsWith("?")){
            return NodeFactory.createVariable(name.replace("?",""));
        }
        else{
            return NodeFactory.createURI(name);
        }
    }

    public Element process(){
        if(this.elementFilter!=null)
            this.elementGroup.addElementFilter(this.elementFilter);
        return this.elementGroup;
    }

    public ElementFilter getElementFilter() {
        return this.elementFilter;
    }

    public ElementGroup getElementGroup() {
        return this.elementGroup;
    }
}
