package gr.uoa.di.ai.gost;

import org.apache.jena.query.Query;
import org.apache.jena.query.SortCondition;
import org.apache.jena.sparql.core.Var;
import org.apache.jena.sparql.core.VarExprList;
import org.apache.jena.sparql.expr.Expr;
import org.apache.jena.sparql.syntax.*;
import org.apache.jena.query.QueryFactory;

import java.util.Map;

public class GoST {

    Query query;

    public GoST(String queryString){
        this.query = QueryFactory.create(queryString);
    }

    public Query processQuery(){
        GeoElementVisitor visitor = new GeoElementVisitor();
        Element element = query.getQueryPattern();
        element.visit(visitor);

        Element block = visitor.generateBlock();


        Query processedQuery = new Query(this.query.getPrologue());

        //set query type
        processedQuery.setQueryPattern(block);
        if(this.query.isSelectType()) {
            processedQuery.setQuerySelectType();

            VarExprList varExprList = this.query.getProject();

            for(String var:this.query.getResultVars()){
                if(varExprList!=null) {
                    Expr aggExpr = varExprList.getExpr(Var.alloc(var));
                    if (aggExpr!=null)
                        processedQuery.addResultVar(var,aggExpr);
                    else
                        processedQuery.addResultVar(var);
                }
                else
                    processedQuery.addResultVar(var);
            }
        }
        else if(this.query.isAskType()) processedQuery.setQueryAskType();
        else if(this.query.isDescribeType()) processedQuery.setQueryDescribeType();

        //set distinct
        if(this.query.isDistinct()) processedQuery.setDistinct(true);
        //set limit
        if(this.query.hasLimit()) processedQuery.setLimit(this.query.getLimit());
        //set group by
        if(this.query.hasGroupBy()){
            Map<Var, Expr> map = this.query.getGroupBy().getExprs();
            for(Var v:this.query.getGroupBy().getVars()){
                if(map.get(v)!=null)
                    processedQuery.addGroupBy(v,this.query.getGroupBy().getExprs().get(v));
                else
                    processedQuery.addGroupBy(v.asNode());
            }
        }
        //set order by
        if(this.query.hasOrderBy()){
            for(SortCondition sortCondition:this.query.getOrderBy()){
                processedQuery.addOrderBy(sortCondition);
            }
        }
        //set having
        if(this.query.hasHaving()){
            for(Expr expr:this.query.getHavingExprs())
                processedQuery.addHavingCondition(expr);
        }

        return processedQuery;
    }

}
