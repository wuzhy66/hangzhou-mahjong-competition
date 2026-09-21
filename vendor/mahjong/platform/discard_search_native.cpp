// 复用规则 AI 的牌形计算；在独立动态库中扩展后处理接口，不改变 RuleBot。
#include "../rule_ai/search_native.cpp"

// 一层显式摸牌树：叶子以 gap - 有效进张/(下一张未知总量) 估计再摸一次后的缺口。
// 只传自手与公开剩余牌量；不含对手真实暗牌和牌墙。
static int legal_ukeire(Hand &h, const Hand &r, int meld, bool ycbk) {
    int gap=missing(h,meld), count=0;
    bool bt=gap==1 && ycbk && baotou(h,meld);
    for(int t=0;t<34;t++) if(r[t]>0 && h[t]<4) {
        h[t]++; int next=missing(h,meld);
        if(next<gap && (next>0 || !ycbk || bt || h[33]==0)) count+=r[t];
        h[t]--;
    }
    return count;
}
extern "C" void rb_shape_one(const int *c,const int *r,int meld,int action,int ycbk,int horizon,double *out,int *branches) {
    std::lock_guard<std::mutex> guard(cache_mutex);
    Hand h,remain;std::copy(c,c+34,h.begin());std::copy(r,r+34,remain.begin());
    h[action]--;int total=0;for(int n:remain)total+=n;
    std::fill(branches,branches+34*5,0);
    int gap=missing(h,meld),ukeire=legal_ukeire(h,remain,meld,ycbk);
    bool bt=baotou(h,meld);
    int64_t numerator=0,gap_sum=0,ukeire_sum=0,win_weight=0;
    for(int t=0;t<34;t++) if(remain[t]>0 && h[t]<4) {
        int weight=remain[t];remain[t]--;h[t]++;
        int offset=t*5;branches[offset]=weight;
        bool win=missing(h,meld)==0 && (!ycbk || bt || h[33]==0);
        if(win) {branches[offset+1]=-1;branches[offset+4]=1;win_weight+=weight;}
        else {
            int bestd=-1,bestgap=99,bestu=-1,bestq=999999;
            for(int a=0;a<34;a++) if(h[a]) {
                h[a]--;int g=missing(h,meld);int u=legal_ukeire(h,remain,meld,ycbk);h[a]++;
                int q=g*(total-1)-(horizon>1?u:0);
                if(q<bestq) {bestq=q;bestgap=g;bestu=u;bestd=a;}
            }
            branches[offset+1]=bestd;branches[offset+2]=bestgap;branches[offset+3]=bestu;
            numerator+=int64_t(weight)*bestq;gap_sum+=int64_t(weight)*bestgap;ukeire_sum+=int64_t(weight)*bestu;
        }
        h[t]--;remain[t]++;
    }
    out[0]=gap;out[1]=ukeire;out[2]=double(numerator)/(total*(total-1));
    out[3]=double(win_weight)/total;out[4]=double(gap_sum)/total;out[5]=double(ukeire_sum)/total;out[6]=total;out[7]=double(numerator);
    if(dfs_cache.size()>2000000)dfs_cache.clear();if(profiles.size()>150000)profiles.clear();
}
