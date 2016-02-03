#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <set>
#include <cmath>
#include <map>
#include <ctime>
#include <dlib/optimization.h>


typedef dlib::matrix<double,0,1> column_vector;

struct dataset
{
	int index;
	std::string sentence;
	std::string prev_val;
	std::string prev_prev_val;
};

//these global variables should be removed ... maybe to return a PAIR
std::vector <dataset> global_d;
std::vector <std::string> global_tags;

std::vector <std::string> global_all_y;

double *global_param;

int (**global_fn_ptr)(int,int);
int global_dim=10;

double global_reg=0;

int global_index; // this variable will have the lesser value between tags , d  ..... used - cost function itertools.izip
int global_num_examples;

std::vector< std::vector<double> > global_dataset;
//std::vector<array<double,20>> global_dataset;   ..... this can be revisted as a std::vector of vectors is not contigous
//map <std::string, double[20]> global_all_data;


/*
if this is removed , create_dataset should return a pair<std::vector<dataset>,std::vector<std::string>>
and thus memm should have these as parameters (std::vector<dataset>,std::vector<std::string>,double)

I am not able to determine wheter to use an array of function pointers or a std::vector of them . Now array is used . If so "memm" should get an extra parameter , number of functions passed . 

The same is with the paramater variable .

and obviously all global variables usage should be changed to parameter variables . Else a MEMM class should be created and those global variables can be instance variables . 

and should global_reg be 0 . Many computations are not needed if so . I know its user input .

written own std::vector dot product as maybe better for parellizing .

train function is not written . As I do not know the DLIB . :(
Can you please do the above 2 ? 


get_features functions needs to be revisited . It solely depends on the input of the user . Now its taking int values .


the constructor __init__ ... is a waste of effort when we are talking about large amount of data .... it blows the memory ... 
imagine 10000 lines 
and all_y has around 30 elements
so around 3 lakh elements in 'dictionary'
i think its better to have a function call every time

even if gradient calls the function that is going to be substituted 3 lakh times .. its better in memory ... if we have inline function then there wont be any change in processing speed also .

your thoughts ??


and train() function is still not done . Can you please see that ??

*/


void split_string(std::string s, std::string* s3 , std::string* s4)
{
	std::string s1,s2;
	int delim=s.find("/");
	//std::cout << delim << endl ;
	int len = s.size();
	//std::cout << len << endl ; 
	s1=s.substr(0,delim);
	s2=s.substr(delim+1,len-1);
	//std::cout << s1 << '\t'<< s1.size() << endl ;
	//std::cout << s2 << '\t' << s2.size() << endl ;
	
	*s3=s1;
	*s4=s2;
	
}

void create_dataset(char *s,char *s1)
{
	std::vector<std::string> untagged ; 
	std::vector<std::string> tags;
	std::vector<dataset> d;
	std::string line,prev,prev_prev ;
	std::string word1,word2;
	int count=0 , word_index =0 , half_index =0,count1=-1;
	std::ifstream myfile,myfile1;

	myfile.open(s);
	if (myfile.is_open())
  	{
    		while ( getline (myfile,line) )
    		{   	
			++count;
			untagged.push_back(line);		
			//std::cout << line << '\n';
    		}
    		myfile.close();
  	}

  	else std::cout << "Unable to open file";


	myfile1.open(s1);
	prev="";
	prev_prev="";
	
	if (myfile1.is_open())
  	{
		
    		while ( getline (myfile1,line) )
    		{		
			
			if(half_index==2)
			{	word_index =0;
				half_index=0;
				prev="";
				prev_prev="";
				++count1;
			}
			if(line=="")
				++half_index;
			
			else
			{
				
				d.push_back(dataset());
				
				d[d.size()-1].sentence=untagged[count1];
               			d[d.size()-1].index=word_index;
				
				if(prev=="")
					d[d.size()-1].prev_val="*";
				else
				{
					split_string(prev,&word1,&word2);
					d[d.size()-1].prev_val=word2;
					
			 	}
			     	if(prev_prev=="")
					d[d.size()-1].prev_prev_val="*";
				else
				{
					split_string(prev_prev,&word1,&word2);
					d[d.size()-1].prev_prev_val=word2;
					
			 	}
			     	++word_index;
				split_string(line,&word1,&word2);
				tags.push_back(word2);
				prev_prev=prev;
				prev=line;				
			}
    		}
    		myfile1.close();
	
	global_d=d;
	global_tags=tags;
	
  	}
  	else std::cout << "Unable to open file";
/*
 	std::cout << d[0].sentence << endl ;
	std::cout << d[0].index << endl ;
	std::cout << d[0].prev_val << endl ;
	std::cout << d[0].prev_prev_val << endl ;
*/
}


int myfunc(int i,int j)
{
	return ( (rand() % (i+1)) + j);
}

/*
*/

double vector_mult(double* a,double* b,int n=10)
{
	int i ;
	double sum=0;
	
	for( i=0 ;i<n;++i)
	{
		sum+=a[i]*b[i];
	}
	
	return sum;

}

void get_features(int x,int y,double* a,int n=10)
{
	
	int i ;
	for(i=0;i<n;i++)
	{
		a[i]=global_fn_ptr[i](x,y);
	}

}

double cost(const column_vector& matrix_vector )
{
	double sum=0,cost,emperical=0,expected =0,dp,temp;
	int i,j;
	int n=10;
	double params[n];
	//std::cout<<"Entering cost\n";
	for (i=0;i<n;i++)
	{
		params[i] = matrix_vector(0,i);

	}
	//global_param=params;
	for (i=0;i<n;++i)
	{
		params[i]=params[i]*params[i];
		sum+=params[i];
	}
	double reg_term= 0.5*global_reg*sum;
	//std::cout<<"Entering main loop\n";
	for(i=0;i<global_index;++i)
	{
		double get_feat[30];
		get_features( (global_d[i]).sentence.size() , (global_tags[i]).size() , get_feat ) ;
		dp=vector_mult( get_feat , params , n);
		emperical+=dp;
		temp=0;
		for(j=0;j<global_all_y.size();++j)
		{
			get_features( (global_d[j]).sentence.size() , (global_all_y[j]).size() , get_feat ) ;
			dp=vector_mult(get_feat , params , n);
			temp+=exp(dp);
		}
		expected += log(temp);
	}
	cost = (expected - emperical) + reg_term ;
	//std::cout<<"cost is\t"<<cost<<std::endl;
	return cost;
}

column_vector gradient(const column_vector& param)
{
	int n=10;
	std::vector <double> gradient ;
	int i,k,j,l;
	double emperical,expected,mysum ,denominator,numerator,div;
	double fx_yprime[20],feats[20];
	double normalizer=0.0;
	double global_param[10];
	for (i=0;i<10;++i)
	{
		global_param[i] = param(0,i);

	}
	for(k=0;k<global_dim;++k)
	{
		emperical = 0.0;
		expected = 0.0;
	
		for(j=0;j<global_dataset.size();j++)
		{
			emperical += global_dataset[j][k];
		}
		
		for(j=0;j<global_num_examples;++j)
		{
			mysum=0.0;
			for(i=0;i<global_all_y.size();++i)
			{
				get_features((global_d[j]).sentence.size() , (global_all_y[i]).size() , fx_yprime);
				
				denominator=0.0;
				numerator=0.0;
				numerator=exp(vector_mult(global_param,fx_yprime));
				
				for(l=0;l<global_all_y.size();++l)
				{
					get_features((global_d[j]).sentence.size() , (global_all_y[i]).size() , feats);	
					denominator+=exp(vector_mult(global_param,fx_yprime));	
				}
				
				div=numerator/denominator;
				
				mysum += div * fx_yprime[k] ;
			}
			
			expected += mysum ;
		}
			gradient.push_back(expected - emperical + global_reg);

	}
	column_vector ret_param(10);
	for (i=0;i<10;i++)
	{
		ret_param(0,i) = gradient[i] ;
		//std::cout<<gradient[i]<<"\t";

	}
	//std::cout<<std::endl;
	return ret_param;

}


void train()
{
	
	
	column_vector starting_point(10);
	starting_point = {0.0};
	clock_t begin = clock();
	dlib::find_min(dlib::lbfgs_search_strategy(100),//TODO: replace hardcoded memory limit 100 by something meaningfull
			dlib::objective_delta_stop_strategy(1e-7,10).be_verbose(),
			cost,gradient,starting_point,-1);
	clock_t end = clock();
	float time_taken = (float) (end-begin)/CLOCKS_PER_SEC;
	std::cout<<"Trained model\n";	
	for(int i=0;i<starting_point.nr();++i)
	{
		std::cout<<starting_point(0,i)<<"\t";
	}
	std::cout<<"\n";
	std::cout<<"Trained in "<<time_taken<<"secs\n";
}

void memm()
{

	int i,j ;
	
	
	for (i=0;i<global_d.size();++i)
	{
	
		double val[20];
		std::vector<double> vals;
		get_features(global_d[i].sentence.size(),global_d[i].sentence.size(),val);
		for(j=0;j<global_dim;++j)
			vals.push_back(val[j]);
		
		global_dataset.push_back(vals);

		//vals.push_back(&val[0],&val[global_dim]);		
	}

}



int main()
{
//	std::string w1,w2;
//	std::string s="hello/world";
//	split_std::string(s,&w1,&w2);
//	std::cout << w1 << endl ;
//	std::cout << w2 << endl ;
	char filename1[25]="tagged_sent_sample";
	char filename2[25]="untagged_sent_sample";
//	create_dataset(filename1);
	create_dataset(filename2,filename1);
	int i;
	double params[10];
//	std::vector <double> params ;
	
	for(i=0;i<10;++i)
	{
		params[i]=0.0 ;
//		params.push_back(0.0);
	}
	
//	function<int(int)> fn_ptr[10];

	int (*fn_ptr[10])(int,int);
	
//	std::vector<int (*)(int,int)> func;
	for(i=0;i<10;++i)
	{
		fn_ptr[i]=&myfunc;
	//	func.push_back(myfunc);
	}
	
	global_fn_ptr=fn_ptr;
	
	/*
	for(i=0;i<10;++i)
	{
		std::cout<<func[i](i,i)<<endl;
	}
	
	for(i=0;i<10;++i)
	{
		std::cout << params[i] << endl;
	}
	*/

	global_num_examples=global_d.size();
	std::set <std::string> all_y;
	global_index=(global_d.size()<global_tags.size())?global_d.size():global_tags.size();
	for(i=0;i<global_tags.size();++i)
	{
		all_y.insert(global_tags[i]);
	}
	global_all_y.assign(all_y.begin(), all_y.end());
	train();
/*
end of run.py 
*/


	return 0;

}


