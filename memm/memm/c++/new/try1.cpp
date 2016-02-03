#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <cmath>
#include <map>
#include <chrono>
#include <cstdlib>
#include <dlib/optimization.h>

#define MAXLEN 3
#define TIME_MEASURE milliseconds
#define MAX_ITERATIONS 110

typedef dlib::matrix<double,0,1> column_vector;

struct dataset
{
	int index;
	std::vector<std::string> sentence ;
	std::string prev_val;
	std::string prev_prev_val;
};

std::vector <dataset> global_d;
std::vector <std::string> global_tags;

std::vector <std::string> global_all_y;

int no_of_calls_cost;
int no_of_calls_gradient;
long int tot_time_cost;
long int tot_time_gradient;
double *global_param;

int (**global_fn_ptr)(int,int);
int global_dim=MAXLEN;

double global_reg=0;

int global_index; 
int global_num_examples;

std::vector< std::vector<double> > global_dataset;

#if 1
std::map <std::string,int> global_tag_index;
std::vector <std::string> global_words;
std::set <std::string> global_all_words;
std::map <std::string,int> global_word_index;
#endif


void split_string(std::string s, std::string* s3 , std::string* s4)
{
	std::string s1,s2;
	int delim=s.find("/");
	int len = s.size();
	s1=s.substr(0,delim);
	s2=s.substr(delim+1,len-1);
	*s3=s1;
	*s4=s2;
	
}
#if 1
void split_string_space(std::string s, std::vector<std::string>& x)
{
	std::string s1,s2;
	int delim,len;
	
	while((delim=s.find(" ")) >= 0)
	{
	len = s.size(); 
	s1=s.substr(0,delim);
	s=s.substr(delim+1,len-1);
	x.push_back(s1);
	
	}
	x.push_back(s);
	
}
#endif
void create_dataset(char *s,char *s1)
{	
	std::vector<std::string> untagged ; 
	std::vector<std::string> tags;
	std::vector<dataset> d;
	std::string line,prev,prev_prev ;
	std::string word1,word2;
	int count=0 , word_index =0 , half_index =0,count1=-1;
	std::ifstream myfile,myfile1;
	#if 1
	int len_sentence ;
	#endif

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
				
				#if 1
				std::vector<std::string> word_sentence ;
				split_string_space(untagged[count1],word_sentence);
				len_sentence=word_sentence.size();
				#endif
				
				d.push_back(dataset());
				
				#if 1
				if(word_index>0)
					d[d.size()-1].sentence.push_back(word_sentence[word_index-1]);
				else
					d[d.size()-1].sentence.push_back("*");

				if(word_index>1)
					d[d.size()-1].sentence.push_back(word_sentence[word_index-2]);
				else
					d[d.size()-1].sentence.push_back("*");

				d[d.size()-1].sentence.push_back(word_sentence[word_index]);

				if(word_index<len_sentence-1)
					d[d.size()-1].sentence.push_back(word_sentence[word_index+1]);
				else
					d[d.size()-1].sentence.push_back("*");

				if(word_index<len_sentence-2)
					d[d.size()-1].sentence.push_back(word_sentence[word_index+2]);
				else
					d[d.size()-1].sentence.push_back("*");

				
//				d[d.size()-1].sentence=untagged[count1];
				#endif
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
				#if 1
				global_words.push_back(word1);
				#endif
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

}

#if 1
void preprocess()
{
	std::map <std::string , int> word_count ;	
	int i,index=0 ;
	std::map<std::string, int>::iterator iter;

	for(i=0;i<global_all_y.size();++i)
	{
		global_tag_index[global_all_y[i]]=index++;
	}

	index=0;

	for(i=0;i<global_words.size();++i)
	{
		word_count[global_words[i]]++;
	}
	
	for(iter=word_count.begin();iter!=word_count.end();++iter)
	{
		if(iter->second)
		{
			global_all_words.insert(iter->first);
			global_word_index[iter->first]=index++;
		}
	}
	//global_dim=word_count.size() * global_all_y.size() ;

}
#endif

int myfunc(int i,int j)
{
	#if 1
	srand(time(NULL));
	double z= ( (double)rand() / RAND_MAX) +0.5;
	return (int)z ;
	#endif
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
#if 1
void get_features(std::string x,std::string y,double* a)
{
	//std::cout<<"X: "<<x<<" Y: "<<y<<"\n";

	if (int(x[0]) >= 65 and int(x[0]) <= 92)	a[0] = 1;
	else 
		a[0] = 0;
	
	if (x.length() >=3 ) a[1] = 1;
	else a[1] = 0;
	if (y.length() <3 )	a[2]=1;
	else a[2] = 0;
	//std::cout<<"THE FEATURES\n";
	//for(int i=0;i<3;++i)	std::cout << a[i] << "\t" ;

	//std::cout << "\n" ;
	
	/*a1 = global_word_index[x];
	a2 = global_tag_index[y] ;
	a[global_all_words.size()*a2 + a1] = 1.0;*/

}
#endif

double cost(const column_vector& matrix_vector)
{
	++no_of_calls_cost;
	if (PRINT_DEBUG_LEVEL == 2) std::cout<<"Cost Call no: "<<no_of_calls_cost<<"\n";	
	auto t1 = std::chrono::high_resolution_clock::now();
	double sum=0,cost,emperical=0,expected =0,dp,temp;
	int i,j;
	double params[MAXLEN];
	for (i=0;i<MAXLEN;i++)
	{
		params[i] = matrix_vector(0,i);
		if (params[i] != params[i]){
			params[i] = 0.0;
			
		}

	}
	global_param=params;
	for (i=0;i<MAXLEN;++i)
	{
		params[i]=params[i]*params[i];
		sum+=params[i];
	}
	double reg_term= 0.5*global_reg*sum;
	for(i=0;i<global_index;++i)
	{
		double get_feat[MAXLEN];
		#if 1
		get_features( (global_d[i]).sentence[2] , global_tags[i] , get_feat ) ;
		#endif
		dp=vector_mult( get_feat , params , MAXLEN);
		emperical+=dp;
		temp=0;
		for(j=0;j<global_all_y.size();++j)
		{
			#if 1
			get_features( (global_d[i]).sentence[2] , global_all_y[j] , get_feat ) ;
			#endif
			dp=vector_mult(get_feat , params , MAXLEN);
			temp+=exp(dp);
		}
		expected += log(temp);

	}
	cost = (expected - emperical) + reg_term ;
	if (std::isnan(cost)) cost=0.0; // re-normalising false cost computation due to input (or sometimes) dlib::error
	auto t2 = std::chrono::high_resolution_clock::now();
	tot_time_cost += std::chrono::duration_cast<std::chrono::TIME_MEASURE> (t2 - t1).count();
	return cost;
}

column_vector gradient(const column_vector& matrix)
{
	++no_of_calls_gradient;	
	if (PRINT_DEBUG_LEVEL == 2) std::cout<<"Gradient Call no: "<<no_of_calls_gradient<<"\n";
	auto t1 = std::chrono::high_resolution_clock::now();
	double param[MAXLEN];
	std::vector <double> gradient ;
	for (int i=0;i<MAXLEN;++i)
	{
		param[i] = matrix(0,i);
		if (std::isnan(param[i])) {param[i] = 0.0;std::cout<<"hi"<<std::endl;}

	}
	if (GRADIENT_VERSION ==3){
	global_param=param;
	
	int i,k,j,l;
	double emperical,expected,mysum ,denominator,numerator,div;
	double fx_yprime[MAXLEN],feats[MAXLEN];
	double normalizer=0.0;
	
	#if 2
	std::vector <double> all_sum;
	int abc;
	#endif

	//std::cout << "dim" << global_dim << "num eg" << global_num_examples << "all_y" << global_all_y.size() << std::endl ;
	for(abc=0;abc<global_dim;++abc)
		all_sum.push_back(0);

	for(j=0;j<global_num_examples;++j)
	{
		denominator=0.0;
		div = 0;		
		for(l=0;l<global_all_y.size();++l)
		{
		#if 1
		get_features((global_d[j]).sentence[2] , global_all_y[l] , feats);	
		#endif					
		denominator+=exp(vector_mult(global_param,feats));
		if (std::isnan(denominator)) denominator = 0.0; // re-normalising error computation based on Ratnaparki-1997 	
		}	
	
		for(i=0;i<global_all_y.size();++i)
		{
		#if 1
		get_features((global_d[j]).sentence[2] , global_all_y[i] , fx_yprime);
		#endif
		numerator=0.0;
		numerator=exp(vector_mult(global_param,fx_yprime));
		if (std::isnan(numerator)) numerator=0.0; // re-normalising error computation based on Ratnaparki-1997 
		div=numerator/denominator;
		
		for(abc=0;abc<global_dim;++abc)
			all_sum[abc] += div * fx_yprime[abc] ;
		
		}
	
	}			
	
	for(k=0;k<global_dim;++k)
	{
		emperical = 0.0;
		expected = 0.0;
		
		for(j=0;j<global_dataset.size();j++)
		{
			emperical += global_dataset[j][k];
		}
		std::cout<<  "emperical " << emperical<< " reg " << global_reg << " expected "<< all_sum[k] << std::endl;
			gradient.push_back(all_sum[k] - emperical + global_reg);
		//std::cout<<"1: " << all_sum[k] + global_reg << " :2: " << all_sum[k]-emperical << std::endl ;
		//std::cout<<"gradient" << all_sum[k]-emperical +global_reg << std::endl ;
	}
	}
	if (GRADIENT_VERSION == 2){	
	global_param=param;
	int i,k,j,l;
	double emperical,expected,mysum ,denominator,numerator,div;
	double fx_yprime[MAXLEN],feats[MAXLEN];
	double normalizer=0.0;
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
			denominator=1.0;
			for(l=0;l<global_all_y.size();++l)
			{
					#if 1
					get_features((global_d[j]).sentence[2] , global_all_y[l] , feats);	
					#endif					
					denominator+=exp(vector_mult(global_param,feats));
					if (std::isnan(denominator)) denominator =0.45; // re-normalising based onRatnaparki-1997 

			}

			mysum=0.0;
			for(i=0;i<global_all_y.size();++i)
			{
				#if 1
				get_features((global_d[j]).sentence[2] , global_all_y[i] , fx_yprime);
				#endif
				
				numerator=0.0;
				numerator=exp(vector_mult(global_param,fx_yprime));
				if (std::isnan(numerator)) numerator = 0.54;// re-normalising error computation based on Ratnaparki-1997 

				div=numerator/denominator;
				
				mysum += div * fx_yprime[k] ;
			}
			
			expected += mysum ;
		}
			gradient.push_back(expected - emperical + global_reg);
	}
	}
	column_vector ret_param(MAXLEN);
	std::cout<<"[\t";
	for (int i=0;i<MAXLEN;++i)
	{
		ret_param(0,i) = gradient[i];
		std::cout<<gradient[i]<<"\t";

	}
	
	auto t2 = std::chrono::high_resolution_clock::now();
	tot_time_gradient += std::chrono::duration_cast<std::chrono::TIME_MEASURE> (t2 - t1).count();
	std::cout<<"\t]\n";
	return ret_param;
}


column_vector train()
{
	preprocess();
	column_vector initial_guess(MAXLEN);
	initial_guess = {0.0};	
	#if 1
	dlib::find_min(dlib::lbfgs_search_strategy(MAXLEN),
			dlib::objective_delta_stop_strategy(1e-07,MAX_ITERATIONS).be_verbose(),
			cost,gradient,initial_guess,-1);
	#endif
	
	//cost(initial_guess);
	//gradient(initial_guess);
	return initial_guess;
}

void memm()
{

	
	
	#if 1
	int i,j,k ;
	for(i=0;i<global_d.size();++i)
	{
		for (j=0;j<global_all_y.size();++j)
		{
			if(global_tags[i]==global_all_y[j])
			{
				double val[MAXLEN];
				std::vector<double> vals;
				#if 1
				get_features(global_d[i].sentence[2] , global_all_y[j] ,val);
				#endif
				for(k=0;k<global_dim;++k)
					vals.push_back(val[k]);
		
				global_dataset.push_back(vals);
			}
		}
	}	


	#endif

	#if 0
	int i,j ;
	for (i=0;i<global_all_y.size();++i)
	{
		/*
			for i,x in enumerate(self.X):
			for y in self.all_y:
				feats = self.all_data.get(y, [])
				val = self.get_features(x, y)
				feats.append(val)
				self.all_data[y] = feats
				if (self.Y[i] == y):
					self.dataset.append(val)

		*/	
		
		double val[MAXLEN];
		std::vector<double> vals;
		#if 1
		get_features(global_d[i].sentence[2] , global_d[i].sentence[2] ,val);
		#endif
		for(j=0;j<global_dim;++j)
			vals.push_back(val[j]);
		
		global_dataset.push_back(vals);
		
		//vals.push_back(&val[0],&val[global_dim]);		
	}
	#endif
}



int main()
{

	char filename1[25]="tagged_sent_sample";
	char filename2[25]="untagged_sent_sample";
	create_dataset(filename2,filename1);
	

	int i;

	int (*fn_ptr[MAXLEN])(int,int);
	
	for(i=0;i<MAXLEN;++i)
	{
		fn_ptr[i]=&myfunc;
	}
	
	global_fn_ptr=fn_ptr;
	
	
	global_num_examples=global_d.size();
	std::set <std::string> all_y;
	global_index=(global_d.size()<global_tags.size())?global_d.size():global_tags.size();
	for(i=0;i<global_tags.size();++i)
	{
		all_y.insert(global_tags[i]);

	}
	global_all_y.assign(all_y.begin(), all_y.end());

	column_vector trained_model(MAXLEN);
	trained_model = {0.0};
	for (int i=0;i<=trained_model.nc()+1;i++) std::cout<<trained_model(0,i)<<"\t";
	memm();
	trained_model = train();
	std::string time_unit = " milliseconds\n";
	if (PRINT_DEBUG_LEVEL >= 1){
	std::cout<<"For "<<no_of_calls_cost<<" calls to cost, total time taken is "<<tot_time_cost<<time_unit;
	std::cout<<"Per call, avg time is "<<(tot_time_cost/no_of_calls_cost)<<time_unit;
	std::cout<<"For "<<no_of_calls_gradient<<" calls to gradient, total time taken is "<<tot_time_gradient<<time_unit;
	std::cout<<"Per call, avg time is "<<(tot_time_gradient/no_of_calls_gradient)<<time_unit;
	}
	std::cout<<"Trained model\n";
	for (int i=0;i<=trained_model.nc()+1;i++) std::cout<<trained_model(0,i)<<"\t";
	std::cout<<"\n";
	
	return 0;

}


