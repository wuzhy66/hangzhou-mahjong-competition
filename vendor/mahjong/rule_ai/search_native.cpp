// 生产规则 AI 的有限步数 DP；所有抽牌都从本家未知牌计数中无放回取样。
#include <array>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
using Hand = std::array<int,34>;
// ctypes.CDLL 调用会释放 GIL；三个入口共用缓存，必须连读、写、清理一起保护。
// 同一进程内串行执行 native 调用，多进程 worker 仍各自并行。
static std::mutex cache_mutex;
static std::unordered_map<uint64_t,int> dfs_cache;
static std::unordered_map<uint64_t,std::array<int,10>> profiles;

static uint64_t pack(const int* c,int n) {
    uint64_t k=0; for(int i=0;i<n;i++) k|=uint64_t(c[i])<<(i*3); return k;
}
static int dfs(std::array<int,9> c,int n,int sets,int pair,bool honor) {
    if(!sets&&!pair) return 0;
    uint64_t key=pack(c.data(),n)|(uint64_t(sets)<<27)|(uint64_t(pair)<<30)|(uint64_t(honor)<<31);
    auto it=dfs_cache.find(key); if(it!=dfs_cache.end()) return it->second;
    int p=0; while(p<n&&!c[p])p++;
    if(p==n) return sets*3+pair*2;
    int v=c[p]; c[p]=0; int best=dfs(c,n,sets,pair,honor); c[p]=v;
    if(pair) for(int k=1;k<=std::min(2,v);k++) {
        c[p]-=k; best=std::min(best,2-k+dfs(c,n,sets,0,honor)); c[p]+=k;
    }
    if(sets) {
        for(int k=1;k<=std::min(3,v);k++) {
            c[p]-=k; best=std::min(best,3-k+dfs(c,n,sets-1,pair,honor)); c[p]+=k;
        }
        if(!honor&&p+2<n) for(int b=0;b<2;b++) for(int d=0;d<2;d++) {
            if(c[p+1]<b||c[p+2]<d)continue;
            c[p]--; c[p+1]-=b;c[p+2]-=d;
            best=std::min(best,2-b-d+dfs(c,n,sets-1,pair,honor));
            c[p]++;c[p+1]+=b;c[p+2]+=d;
        }
        if(!honor&&p>0&&p+1<n&&c[p+1]) {
            c[p]--;c[p+1]--; best=std::min(best,1+dfs(c,n,sets-1,pair,honor));c[p]++;c[p+1]++;
        }
    }
    dfs_cache[key]=best;return best;
}
static std::array<int,10> profile(const int* c,int n,bool honor) {
    uint64_t key=pack(c,n)|(uint64_t(honor)<<31);
    auto it=profiles.find(key); if(it!=profiles.end())return it->second;
    std::array<int,9>a{};std::copy(c,c+n,a.begin());std::array<int,10>r;
    for(int s=0;s<=4;s++)for(int p=0;p<2;p++)r[2*s+p]=dfs(a,n,s,p,honor);
    profiles[key]=r;return r;
}
static int qmissing(const Hand&c,int meld) {
    if(meld)return 99;
    int pairs=0,singles=0,god=c[33];
    for(int i=0;i<33;i++){pairs+=c[i]/2;singles+=c[i]%2;}
    int need=std::max(0,7-pairs),x=std::min({need,singles,god});
    need-=x;singles-=x;god-=x;x=std::min(need,god/2);need-=x;god-=2*x;
    return 2*need-std::min(need,singles+god);
}
static int missing(const Hand&c,int meld) {
    int need=4-meld;std::array<int,10>dp;dp.fill(99);dp[0]=0;
    for(int g=0;g<4;g++) {
        auto pr=profile(c.data()+g*9,g==3?6:9,g==3);
        std::array<int,10>next;next.fill(99);
        for(int s=0;s<=need;s++)for(int p=0;p<2;p++)for(int a=0;a+s<=need;a++)for(int b=0;b+p<2;b++)
            next[2*(s+a)+p+b]=std::min(next[2*(s+a)+p+b],dp[2*s+p]+pr[2*a+b]);
        dp=next;
    }
    return std::min(std::max(0,dp[2*need+1]-c[33]),qmissing(c,meld));
}
static bool baotou(Hand&c,int meld) {
    if(missing(c,meld)!=1)return false;
    for(int t=0;t<34;t++)if(c[t]<4) {
        c[t]++;bool win=missing(c,meld)==0;c[t]--;if(!win)return false;
    }
    return true;
}
static double fan(const Hand&c,int meld,bool bt,int chain,int piao) {
    int factor=1;
    if(qmissing(c,meld)==0) {
        int quads=0,singles=0;for(int t=0;t<33;t++){quads+=c[t]==4;singles+=c[t]%2;}
        quads+=(c[33]-std::min(c[33],singles))==4;
        factor=1<<(1+std::min(3,quads));
    }
    return factor*(1<<std::min(6,chain))*(c[33]+piao==4?2:1)*(bt?2:1);
}
struct Value {double score=0,win=0;};
struct Search {
    int meld;double unit,win_bonus,survival;
    bool ycbk,improvements;
    std::unordered_map<std::string,Value>cache;
    uint64_t nodes=0;
    double utility(Value v){return v.score+win_bonus*v.win;}
    Value solve(Hand&h,Hand&r,int depth,int chain=0,int piao=0) {
        if(depth<=0||missing(h,meld)>depth)return {};
        std::string key;key.reserve(72);
        for(int x:h)key.push_back(char(x));for(int x:r)key.push_back(char(x));
        key.push_back(char(depth));key.push_back(char(chain));key.push_back(char(piao));
        auto it=cache.find(key);if(it!=cache.end())return it->second;
        nodes++;int total=0;for(int x:r)total+=x;if(!total)return {};
        bool bt=baotou(h,meld);int oldgap=missing(h,meld);Value out;
        for(int t=0;t<34;t++)if(r[t]>0&&h[t]<4) {
            double prob=double(r[t])/total;r[t]--;h[t]++;
            int gap=missing(h,meld);Value best;
            if(gap==0&&(!ycbk||bt||h[33]==0))best={unit*fan(h,meld,bt,chain,piao),1};
            if(depth>1 && (improvements || gap<oldgap)) {
                // 爆头、七对与普通胡共用终局计分；继续打牌才会更新/断开动作链。
                for(int d=0;d<34;d++)if(h[d]>0) {
                    h[d]--;int nc=d==33&&bt?chain+1:0,np=d==33&&bt?piao+1:0;
                    Value v=solve(h,r,depth-1,nc,np);h[d]++;
                    if(utility(v)>utility(best))best=v;
                }
            } else if(depth>1) {
                // 消融：非进张只允许摸切；仍保留等待下一巡的概率质量。
                h[t]--;best=solve(h,r,depth-1,t==33&&bt?chain+1:0,t==33&&bt?piao+1:0);h[t]++;
            }
            h[t]--;r[t]++;out.score+=prob*best.score;out.win+=prob*best.win;
        }
        out.score*=survival;out.win*=survival;cache.emplace(std::move(key),out);return out;
    }
};
extern "C" {
int rb_missing(const int*c,int meld) {
    std::lock_guard<std::mutex> guard(cache_mutex);
    Hand h;std::copy(c,c+34,h.begin());return missing(h,meld);
}
int rb_discard(const int*c,const int*r,int meld,const int*allowed) {
    std::lock_guard<std::mutex> guard(cache_mutex);
    Hand h;std::copy(c,c+34,h.begin());int low=99;
    std::array<int,34>gaps;gaps.fill(99);
    for(int t=0;t<34;t++)if(allowed[t]&&h[t]) {
        h[t]--;gaps[t]=missing(h,meld);low=std::min(low,gaps[t]);h[t]++;
    }
    std::array<int,4>best{{-1,-1,-1,-1}};int tile=-1;
    for(int t=0;t<34;t++)if(allowed[t]&&h[t]&&gaps[t]==low) {
        h[t]--;int gain=0;bool bt=low==1;
        for(int d=0;d<34;d++)if(h[d]<4) {
            h[d]++;int m=missing(h,meld);h[d]--;
            if(m<low)gain+=r[d];if(m!=0)bt=false;
        }
        std::array<int,4>key{{int(bt),int(t!=33),gain,t}};
        if(key>best){best=key;tile=t;}h[t]++;
    }
    return tile;
}
// 返回 score/win/nodes；本函数只看传入的手牌与未知牌计数，没有引擎暗牌入口。
void rb_value(const int*c,const int*r,int meld,int depth,int chain,int piao,double unit,double bonus,double survival,int ycbk,int improve,double*out) {
    std::lock_guard<std::mutex> guard(cache_mutex);
    Hand h,remain;std::copy(c,c+34,h.begin());std::copy(r,r+34,remain.begin());
    Search s{meld,unit,bonus,survival,bool(ycbk),bool(improve),{},0};
    Value v=s.solve(h,remain,depth,chain,piao);out[0]=v.score;out[1]=v.win;out[2]=double(s.nodes);
    // 缓存仅保存形状计算，限制长期评测 worker 的常驻内存。
    if(dfs_cache.size()>2000000)dfs_cache.clear();if(profiles.size()>150000)profiles.clear();
}
}
